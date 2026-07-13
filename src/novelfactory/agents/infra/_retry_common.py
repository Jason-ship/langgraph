"""Common retry helpers shared between synchronous and async retry implementations.

This module extracts duplicated logic from retry.py and async_retry.py
to reduce ~120 lines of duplicated code and prevent sync/async drift.
"""

from __future__ import annotations

import asyncio
from typing import Any

from novelfactory.agents.infra.logger import get_logger
from novelfactory.agents.infra.timeout import LLMTimeoutError
from novelfactory.agents.infra.usage import (
    _DEEPSEEK_COMPLETION_RATE,
    _DEEPSEEK_PROMPT_RATE,
    _TOKENS_PER_MILLION,
    _extract_usage_from_result,
    _record_usage,
)

# ── RetryPolicy 映射（sync/async 共享，避免双份定义 drift）─────────────────
from novelfactory.config.constants import (
    DEFAULT_RETRY,
    REVIEWER_RETRY,
    WRITER_RETRY,
)

_RETRY_POLICY_MAP: dict[str, Any] = {
    "default": DEFAULT_RETRY,
    "writer": WRITER_RETRY,
    "reviewer": REVIEWER_RETRY,
}


def _resolve_retry_policy(policy_name: str = "default") -> Any:
    """根据策略名返回 RetryPolicy 配置。"""
    return _RETRY_POLICY_MAP.get(policy_name, DEFAULT_RETRY)


# ── Error Classification (sync/async 统一) ────────────────────────────────────


def _classify_error(exc: Exception) -> tuple[str, int | None]:
    """Classify an exception to determine retry strategy.

    统一处理 sync 和 async 异常：
      - httpx.HTTPStatusError → 按 HTTP status code 分类
      - LLMTimeoutError / asyncio.TimeoutError / OSError / IOError → backoff
      - 其他 → 尝试提取 HTTP status，否则 backoff
    """
    import httpx

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return _categorize_status(status), status

    # asyncio.TimeoutError 在 Python 3.11+ 是 TimeoutError 的别名 (OSError 子类)，
    # 但在 3.10 中是独立类，需显式检查。
    if isinstance(exc, LLMTimeoutError | asyncio.TimeoutError | OSError | IOError):
        return "backoff", None

    status = _extract_http_status(exc)
    if status is not None:
        return _categorize_status(status), status

    return "backoff", None


# ── Model Name Extraction ────────────────────────────────────────────────────


def _extract_model_name(func: Any, *args: Any, **kwargs: Any) -> str:
    """Extract the model name from a LangChain callable.

    Handles ``RunnableWithFallbacks`` (from ``.with_fallbacks()``) which does
    not expose ``model`` directly — we drill into ``.runnable`` to find the
    primary model name.
    """
    if hasattr(func, "model") and func.model:
        return func.model
    # RunnableWithFallbacks 包装的 primary runnable
    if (
        hasattr(func, "runnable")
        and hasattr(func.runnable, "model")
        and func.runnable.model
    ):
        return func.runnable.model
    if args and hasattr(args[0], "model") and args[0].model:
        return args[0].model
    if hasattr(func, "__self__") and hasattr(func.__self__, "model"):
        return getattr(func.__self__, "model", "") or "deepseek-v4-flash"
    return kwargs.get("model", "deepseek-v4-flash")


# ── HTTP Status Categorization ────────────────────────────────────────────────


def _categorize_status(status: int) -> str:
    """Categorize an HTTP status code into a retry strategy category."""
    from novelfactory.config.constants import (
        NO_RETRY_ON_HTTP_STATUS,
        RETRY_IMMEDIATE_HTTP_STATUS,
        RETRY_ON_HTTP_STATUS,
    )

    if status in NO_RETRY_ON_HTTP_STATUS:
        return "no_retry"
    if status in RETRY_ON_HTTP_STATUS:
        return "backoff"
    if status == RETRY_IMMEDIATE_HTTP_STATUS:
        return "immediate"
    return "backoff"


def _extract_http_status(exc: Exception) -> int | None:
    """Extract HTTP status code from an arbitrary exception.

    Tries common attribute patterns used by various HTTP libraries.
    Returns None if no status code can be extracted.
    """
    status = getattr(exc, "status_code", None)
    if status is not None:
        return int(status)
    response = getattr(exc, "response", None)
    if response is not None:
        return int(getattr(response, "status_code", 0)) or None
    return None


# ── Truncation Detection ──────────────────────────────────────────────────────


def _check_truncation(result: Any) -> bool:
    """Check if the LLM result was truncated due to max_tokens (finish_reason=length).

    Returns True if truncated.
    """
    if not isinstance(result, dict):
        return False
    msgs = result.get("messages", [])
    if not msgs:
        return False
    last_msg = msgs[-1]
    resp_meta = getattr(last_msg, "response_metadata", None) or {}
    usage = getattr(last_msg, "usage_metadata", None) or {}
    stop_reason = (
        resp_meta.get("finish_reason")
        or resp_meta.get("stop_reason")
        or usage.get("finish_reason")
    )
    return bool(stop_reason == "length")


# ── Usage Logging and Audit ───────────────────────────────────────────────────


def _log_usage_and_audit(
    step_name: str, result: Any, call_start: float, model_name: str
) -> None:
    """Log token usage and audit trail for an LLM call.

    v5.4-fix: API 不返回 usage 时从消息文本本地统计 token 数。
    """
    import time

    p_tokens, c_tokens = _extract_usage_from_result(result)
    if not p_tokens and not c_tokens:
        # 从消息文本本地统计（ARK API 不返回标准 usage metadata）
        from novelfactory.agents.infra.usage import count_tokens

        msgs = result.get("messages", []) if isinstance(result, dict) else []
        for msg in msgs:
            content = getattr(msg, "content", None) or (
                msg.get("content") if isinstance(msg, dict) else ""
            )
            if content:
                role = getattr(msg, "type", None) or msg.get("role", "")
                tokens = count_tokens(str(content))
                if role == "assistant":
                    c_tokens += tokens
                else:
                    p_tokens += tokens
        # 兜底: 假设至少有一些 token
        if not p_tokens and not c_tokens:
            return
    _record_usage(step_name, p_tokens, c_tokens, model=model_name)
    duration_ms = (time.time() - call_start) * 1000
    cost = (p_tokens / _TOKENS_PER_MILLION) * _DEEPSEEK_PROMPT_RATE + (
        c_tokens / _TOKENS_PER_MILLION
    ) * _DEEPSEEK_COMPLETION_RATE
    logger = get_logger("novelfactory.llm")
    try:
        from novelfactory.utils.monitoring import AuditLogger

        AuditLogger().log_llm_call(
            model=model_name,
            phase=step_name,
            input_tokens=p_tokens,
            output_tokens=c_tokens,
            duration_ms=round(duration_ms, 2),
            cost_cny=round(cost, 6),
        )
    except ImportError:
        logger.debug("[%s] AuditLogger not available, skipped", step_name)
    logger.info(
        "[%s] usage: prompt=%d completion=%d total=%d cost=¥%.4f",
        step_name,
        p_tokens,
        c_tokens,
        p_tokens + c_tokens,
        cost,
    )


# ── Backoff Delay Calculation (sync/async 共享) ──────────────────────────────


def compute_backoff_delay(
    attempt: int,
    initial_interval: float = 1.0,
    max_interval: float = 60.0,
) -> float:
    """Compute exponential backoff delay with a cap.

    Formula: min(initial_interval * 2^(attempt-1), max_interval)
    Used by both sync (retry.py) and async (async_retry.py) retry loops.
    """
    return min(initial_interval * (2 ** (attempt - 1)), max_interval)


# ── Sentinel Dict Factories (sync/async 共享) ────────────────────────────────


def _build_degraded_return(reason: str = "") -> dict[str, Any]:
    """Build the sentinel dict returned when circuit breaker is open.

    Downstream code detects the ``_degraded`` marker and handles graceful
    fallback instead of treating it as a valid LLM response.
    """
    return {
        "messages": [],
        "crew_result": {},
        "_degraded": True,
        "_degraded_reason": reason,
    }


def _build_exhausted_return(step_name: str) -> dict[str, Any]:
    """Build the sentinel dict returned when all retry attempts are exhausted."""
    return {
        "messages": [],
        "crew_result": {},
        "_degraded": True,
        "_degraded_reason": f"all_retries_exhausted:{step_name}",
    }


# ── Pre-flight Checks (sync/async 共享) ─────────────────────────────────────


def _check_circuit_breakers(
    step_name: str,
    fallback: Any,
) -> tuple[bool, Any | None]:
    """Check circuit breakers for ARK and DeepSeek providers before a call.

    Returns ``(blocked, result)`` where:
      - ``blocked=True``, ``result`` is the degraded/fallback value to return immediately.
      - ``blocked=False``, ``result`` is ``None`` (proceed with the LLM call).

    v5.9 FIX: checks both ARK (primary endpoint) and DeepSeek (first fallback).
    Previously only DeepSeek was checked, leaving ARK unprotected.
    """
    from novelfactory.agents.infra.circuit_breaker import circuit_breaker_is_open

    _logger = get_logger("novelfactory.llm")
    for provider in ("ark", "deepseek"):
        if circuit_breaker_is_open(provider):
            _logger.warning(
                "[%s] Circuit breaker OPEN for %s — fast-failing",
                step_name,
                provider,
            )
            if fallback is None:
                return True, _build_degraded_return(f"circuit_breaker_open:{provider}")
            return True, fallback
    return False, None


def _check_quota_preflight(
    step_name: str,
    fallback: Any,
) -> tuple[bool, Any | None]:
    """Check API quota before making an LLM call.

    Returns ``(blocked, result)`` where:
      - ``blocked=True``, ``result`` is the fallback value to return immediately.
      - ``blocked=False``, ``result`` is ``None`` (proceed with the LLM call).
    """
    from novelfactory.agents.infra.quota import (
        _check_quota_safe,
        _quota_check_before_call,
    )

    _logger = get_logger("novelfactory.llm")
    if _quota_check_before_call():
        blocked, reason = _check_quota_safe()
        if blocked:
            _logger.error("[%s] BLOCKED — %s", step_name, reason)
            return True, fallback
    return False, None


def _record_provider_failures() -> None:
    """Record a failure against both ARK and DeepSeek circuit breakers.

    Called on every LLM error to track provider health for the fallback chain.
    v7.3: 阈值已从 5→20 提升，cooldown 从 120s→30s 缩短，
    正常 transient error 不会触发熔断。
    """
    from novelfactory.agents.infra.circuit_breaker import circuit_breaker_record_failure

    circuit_breaker_record_failure("ark")
    circuit_breaker_record_failure("deepseek")


def _record_provider_success() -> None:
    """Record success against both ARK and DeepSeek circuit breakers."""
    from novelfactory.agents.infra.circuit_breaker import circuit_breaker_record_success

    circuit_breaker_record_success("ark")
    circuit_breaker_record_success("deepseek")
