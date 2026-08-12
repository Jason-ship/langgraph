"""LLM configuration for NovelFactory multi-agent system.

Tech stack: DeepSeek V4 Flash (via DeepSeek 官方 API) — 2026-08-12.
API Endpoint: https://api.deepseek.com

Auto-fallback: DeepSeek 直连失败时降级到硅基流动（仅当配置 API key）。
火山引擎方舟 (ARK) 已停用（v8.2，配额用尽），不再作为 primary 或 fallback。

Single-model strategy (DeepSeek V4 Flash is fast + capable for all tiers):
  - supervisor_llm  : deepseek-v4-flash (temp=0.3)  — high-quality orchestration
  - worker_llm      : deepseek-v4-flash (temp=0.7)  — fast creative work
  - reviewer_llm    : deepseek-v4-flash (temp=0.2)  — structured scoring
  - review_llm      : deepseek-v4-flash (temp=0.2)  — human-in-the-loop final review
  - writing_llm     : deepseek-v4-flash (temp=0.75) — vivid narrative

DeepSeek V4 Flash Context Window: 1M tokens.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI

from novelfactory.config.llm_params import center as llm_param_center
from novelfactory.config.settings import settings

logger = logging.getLogger(__name__)


# ── Fallback sentinel ──────────────────────────────────────────────────────
# Registry so monitoring / health can check fallback status
_fallback_registry: dict[str, bool] = {}
_fallback_timestamps: dict[str, str] = {}  # tier_name → ISO timestamp


def get_fallback_status() -> dict[str, Any]:
    """Return per-tier fallback status for dashboard/monitoring."""
    return {
        "tiers": dict(_fallback_registry),
        "timestamps": dict(_fallback_timestamps),
        "any_active": any(v for v in _fallback_registry.values()),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Fallback tracker wrapper ──────────────────────────────────────────────
# When with_fallbacks() invokes a fallback LLM, set _fallback_registry.


def _wrap_fallback_llm(llm: ChatOpenAI, tier_name: str) -> ChatOpenAI:
    """Monkey-patch a ChatOpenAI instance so its invoke/ainvoke set the fallback flag.

    LangChain's RunnableWithFallbacks calls ``fallback.invoke()`` when the primary
    fails.  Uses ``object.__setattr__`` to bypass Pydantic v2 attribute validation
    (ChatOpenAI is a Pydantic model, and ``invoke`` is not a declared field).

    注意：此方法通过 monkey-patch 实现 fallback 追踪，存在以下脆弱性：
    - 依赖 LangChain ChatOpenAI 内部属性访问机制
    - LangChain 大版本升级可能导致接口不兼容
    - 不影响主调用链路，仅在 fallback LLM 被触发时生效
    - 此方式是 LangChain RunnableWithFallbacks 无回调机制的妥协方案
    """
    original_invoke = llm.invoke

    def _tracked_invoke(input, config=None, **kwargs):
        _fallback_registry[tier_name] = True
        _fallback_timestamps[tier_name] = datetime.now(timezone.utc).isoformat()
        logger.info("[fallback] %s: using fallback endpoint", tier_name)
        return original_invoke(input, config=config, **kwargs)

    # Use object.__setattr__ to bypass Pydantic v2 field validation
    object.__setattr__(llm, "invoke", _tracked_invoke)

    # Also wrap ainvoke for async callers
    if hasattr(llm, "ainvoke"):
        original_ainvoke = llm.ainvoke

        async def _tracked_ainvoke(input, config=None, **kwargs):
            _fallback_registry[tier_name] = True
            _fallback_timestamps[tier_name] = datetime.now(timezone.utc).isoformat()
            logger.info("[fallback] %s: using fallback (async)", tier_name)
            return await original_ainvoke(input, config=config, **kwargs)

        object.__setattr__(llm, "ainvoke", _tracked_ainvoke)

    return llm


# ── Shared configs ─────────────────────────────────────────────────────────


def _get_deepseek_config() -> dict[str, Any]:
    """Return ChatOpenAI kwargs for DeepSeek direct API (fallback endpoint).

    v6.1: 从 settings 读取为主，os.environ 兜底。
    """
    api_key = (
        getattr(settings, "DEEPSEEK_API_KEY", "")
        or settings.ARK_API_KEY
        or os.environ.get("OPENAI_API_KEY")
    )

    return {
        "api_key": api_key,
        "base_url": "https://api.deepseek.com",
        "max_tokens": 65536,
        "request_timeout": settings.LLM_REQUEST_TIMEOUT,
        "max_retries": settings.MAX_RETRIES,
    }


def _get_siliconflow_config() -> dict[str, Any]:
    """Return ChatOpenAI kwargs for 硅基流动 API (second fallback)."""
    api_key = os.environ.get("SILICONFLOW_API_KEY") or getattr(
        settings, "SILICONFLOW_API_KEY", ""
    )

    return {
        "api_key": api_key,
        "base_url": getattr(settings, "SILICONFLOW_BASE_URL", ""),
        "max_tokens": 65536,
        "request_timeout": settings.LLM_REQUEST_TIMEOUT,
        "max_retries": 3,
    }


# ── Fallback factory ───────────────────────────────────────────────────────


def _create_with_auto_fallback(
    temperature: float,
    tier_name: str = "",
) -> ChatOpenAI:
    """Create a ChatOpenAI with optional auto-fallback (v8.2: DeepSeek 直连为主).

    Fallback chain:
      1. Primary: DeepSeek 官方 API (deepseek-v4-flash)
      2. Fallback: 硅基流动 API (deepseek-v4-flash, 仅当 API key 配置时启用)

    火山引擎方舟 (ARK) 已于 v8.2 停用（配额用尽），不再作为 primary 或 fallback。
    配置了 SILICONFLOW_API_KEY 时启用二级降级，否则单路直连。
    """
    deepseek_config = _get_deepseek_config()

    primary = ChatOpenAI(
        model="deepseek-v4-flash",
        temperature=temperature,
        **deepseek_config,
    )

    fallbacks: list[ChatOpenAI] = []

    # 硅基流动降级可选 — 仅在配置了 API key 时启用
    siliconflow_api_key = getattr(settings, "SILICONFLOW_API_KEY", "") or os.environ.get(
        "SILICONFLOW_API_KEY"
    )
    if siliconflow_api_key:
        siliconflow_config = _get_siliconflow_config()
        fallback1 = ChatOpenAI(
            model="deepseek-v4-flash",
            temperature=temperature,
            **siliconflow_config,
        )
        _wrap_fallback_llm(fallback1, tier_name) if tier_name else None
        fallbacks.append(fallback1)

    llm = primary.with_fallbacks(fallbacks) if fallbacks else primary

    # Register for monitoring
    if tier_name:
        _fallback_registry[tier_name] = False
        key_count = 2 if siliconflow_api_key else 1
        logger.debug(
            "[fallback] %s: primary=DeepSeek(f1=硅基流动) x%s",
            tier_name,
            key_count,
        )

    return llm


# ── Tiered LLM factories (all with auto-fallback) ──────────────────────────


@lru_cache(maxsize=1)
def get_supervisor_llm() -> ChatOpenAI:
    p = llm_param_center.get_params("supervisor")
    return _create_with_auto_fallback(temperature=p.temperature, tier_name="supervisor")


@lru_cache(maxsize=1)
def get_worker_llm() -> ChatOpenAI:
    p = llm_param_center.get_params("worker")
    return _create_with_auto_fallback(temperature=p.temperature, tier_name="worker")


@lru_cache(maxsize=1)
def get_reviewer_llm() -> ChatOpenAI:
    p = llm_param_center.get_params("reviewer")
    return _create_with_auto_fallback(temperature=p.temperature, tier_name="reviewer")


@lru_cache(maxsize=1)
def get_review_llm() -> ChatOpenAI:
    p = llm_param_center.get_params("review")
    return _create_with_auto_fallback(temperature=p.temperature, tier_name="review")


@lru_cache(maxsize=1)
def get_writing_llm() -> ChatOpenAI:
    p = llm_param_center.get_params("writing")
    return _create_with_auto_fallback(temperature=p.temperature, tier_name="writing")


def reset_llm_cache() -> None:
    """清除 LLM 工厂缓存（温度等参数变更后调用）。

    当 QualityParameterCenter / LLMParameterCenter 修改了温度参数时，
    必须清除 @lru_cache 缓存的 LLM 实例，否则新参数不生效。
    """
    get_supervisor_llm.cache_clear()
    get_worker_llm.cache_clear()
    get_reviewer_llm.cache_clear()
    get_review_llm.cache_clear()
    get_writing_llm.cache_clear()
    logger.info("[LLM] All LLM factory caches cleared")
