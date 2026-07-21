"""LLM configuration for NovelFactory multi-agent system.

Tech stack: DeepSeek V4 Flash (via 火山引擎方舟 Coding Plan) — 2026-06-17.
API Endpoint: https://ark.cn-beijing.volces.com/api/coding/v3

Auto-fallback: if ARK returns rate-limit / server errors, automatically
downgrades to DeepSeek direct API (api.deepseek.com / deepseek-chat).

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


def _get_ark_config() -> dict[str, Any]:
    """Return ChatOpenAI kwargs for 火山引擎方舟 Coding Plan endpoint.

    v6.1: 从 settings 读取为主，os.environ 兜底。
    """
    api_key = settings.ARK_API_KEY or None
    if not api_key:
        api_key = getattr(settings, "DEEPSEEK_API_KEY", "") or os.environ.get(
            "OPENAI_API_KEY"
        )

    base_url = (
        settings.ARK_BASE_URL or "https://ark.cn-beijing.volces.com/api/coding/v3"
    )

    return {
        "api_key": api_key,
        "base_url": base_url,
        "max_tokens": 65536,
        "request_timeout": settings.LLM_REQUEST_TIMEOUT,
        "max_retries": settings.MAX_RETRIES,
    }


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
    """Create a ChatOpenAI with multi-layer auto-fallback (v5.1.1: 硅基流动可选降级).

    Fallback chain:
      1. Primary: 火山引擎方舟 ARK (deepseek-v4-flash)
      2. Fallback: DeepSeek 官方 API (deepseek-v4-flash)
      3. Fallback: 硅基流动 API (deepseek-v4-flash, 仅当 API key 配置时启用)

    配置了 SILICONFLOW_API_KEY 时启用三级降级，否则只有两级降级。
    """
    ark_config = _get_ark_config()
    deepseek_config = _get_deepseek_config()

    primary = ChatOpenAI(
        model="deepseek-v4-flash",
        temperature=temperature,
        **ark_config,
    )

    fallback1 = ChatOpenAI(
        model="deepseek-v4-flash",
        temperature=temperature,
        **deepseek_config,
    )
    _wrap_fallback_llm(fallback1, tier_name) if tier_name else None

    fallbacks = [fallback1]

    # 硅基流动降级可选 — 仅在配置了 API key 时启用
    siliconflow_api_key = getattr(settings, "SILICONFLOW_API_KEY", "") or os.environ.get(
        "SILICONFLOW_API_KEY"
    )
    if siliconflow_api_key:
        siliconflow_config = _get_siliconflow_config()
        fallback2 = ChatOpenAI(
            model="deepseek-v4-flash",
            temperature=temperature,
            **siliconflow_config,
        )
        _wrap_fallback_llm(fallback2, tier_name) if tier_name else None
        fallbacks.append(fallback2)

    llm = primary.with_fallbacks(fallbacks)

    # Register for monitoring
    if tier_name:
        _fallback_registry[tier_name] = False
        key_count = 3 if siliconflow_api_key else 2
        logger.debug(
            "[fallback] %s: primary=ARK(f1=DeepSeek(f2=硅基流动) x%s)",
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
