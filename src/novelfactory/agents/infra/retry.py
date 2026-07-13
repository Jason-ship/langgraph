"""LLM call with timeout + error-type-aware retry and usage auditing.

Synchronous version. For the async counterpart, see ``async_retry.py``.
Both share helpers defined in ``_retry_common.py``.
"""

from __future__ import annotations

import time
from typing import Any

from novelfactory.agents.infra._retry_common import (
    _build_exhausted_return,
    _check_circuit_breakers,
    _check_quota_preflight,
    _check_truncation,
    _classify_error,
    _extract_model_name,
    _log_usage_and_audit,
    _record_provider_failures,
    _record_provider_success,
    _resolve_retry_policy,
    compute_backoff_delay,
)
from novelfactory.agents.infra.logger import get_logger
from novelfactory.agents.infra.timeout import LLMTimeoutError, with_timeout
from novelfactory.agents.infra.usage import (
    _extract_usage_from_result,
    _record_usage,
)
from novelfactory.config.constants import (
    DEFAULT_TIMEOUT,
    TIMEOUT_EXTRACT,
    TIMEOUT_LONG,
    TIMEOUT_SHORT,
)

# ── Retry Configuration ───────────────────────────────────────────────────────

_DEFAULT_TIMEOUT = DEFAULT_TIMEOUT
TIMEOUT_EXTRACT = TIMEOUT_EXTRACT
TIMEOUT_SHORT = TIMEOUT_SHORT
TIMEOUT_LONG = TIMEOUT_LONG


def llm_call_with_retry(
    func: Any,
    *args: Any,
    step_name: str = "llm_call",
    retry_policy: str = "default",
    timeout_seconds: float = _DEFAULT_TIMEOUT,
    fallback: Any = None,
    **kwargs: Any,
) -> Any:
    """Call an LLM function with timeout guard + error-type-aware retry.

    使用 graph/checkpointer.py 定义的 RetryPolicy 常量控制重试行为。
    retry_policy: "default" | "writer" | "reviewer"

    Retry strategy:
      - 400/401/403: never retry
      - 429: immediate retry
      - 500/502/503/504: exponential backoff
      - TimeoutError/OSError: exponential backoff

    Side effect: appends token usage to thread-local accumulator.
    """
    _call_start = time.time()
    logger = get_logger("novelfactory.llm")
    policy = _resolve_retry_policy(retry_policy)
    max_retries = policy.max_attempts

    # Pre-flight: quota check
    blocked, result = _check_quota_preflight(step_name, fallback)
    if blocked:
        return result

    # Pre-flight: circuit breaker fast-fail (v5.9: both ARK and DeepSeek)
    blocked, result = _check_circuit_breakers(step_name, fallback)
    if blocked:
        return result

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            timed_func = with_timeout(timeout_seconds, None)(func)
            result = timed_func(*args, **kwargs)

            if result is None:
                _msg = f"{step_name} timed out after {timeout_seconds}s"
                raise LLMTimeoutError(_msg)

            # Check for finish_reason=length truncation
            truncated = _check_truncation(result)
            if truncated:
                logger.warning(
                    "[%s] attempt %d hit max_tokens ceiling (finish_reason=length). Retrying.",
                    step_name,
                    attempt,
                )
            if truncated and attempt < max_retries:
                p_tokens, c_tokens = _extract_usage_from_result(result)
                model_name = _extract_model_name(func, *args, **kwargs)
                if p_tokens or c_tokens:
                    _record_usage(step_name, p_tokens, c_tokens, model=model_name)
                continue

            if attempt > 1:
                logger.info("[%s] succeeded on attempt %d", step_name, attempt)

            msgs = result.get("messages", []) if isinstance(result, dict) else []
            logger.info("[%s] done — %d messages", step_name, len(msgs))

            model_name = _extract_model_name(func, *args, **kwargs)
            _log_usage_and_audit(step_name, result, _call_start, model_name)

            _record_provider_success()
            return result

        # 重试循环需要捕获所有异常类型以决定是否重试；_classify_error 会过滤不可重试的错误
        except Exception as exc:
            last_err = exc
            category, http_status = _classify_error(exc)

            if category == "no_retry":
                logger.error(
                    "[%s] attempt %d/%d — %s (HTTP %s) — NOT retrying",
                    step_name,
                    attempt,
                    max_retries,
                    type(exc).__name__,
                    http_status or "unknown",
                )
                break

            if category == "immediate":
                logger.warning(
                    "[%s] attempt %d/%d — rate limited (HTTP %s) — retrying immediately",
                    step_name,
                    attempt,
                    max_retries,
                    http_status or 429,
                )
                _record_provider_failures()
                if attempt < max_retries:
                    continue

            _record_provider_failures()

            delay = compute_backoff_delay(
                attempt,
                policy.initial_interval,
                getattr(policy, "max_interval", 60.0),
            )
            logger.warning(
                "[%s] attempt %d/%d failed (%s, HTTP %s) — retrying in %ds",
                step_name,
                attempt,
                max_retries,
                type(exc).__name__,
                http_status or "unknown",
                delay,
            )
            if attempt < max_retries:
                time.sleep(delay)

    if last_err:
        logger.error(
            "[%s] failed after %d attempts: %s", step_name, max_retries, last_err
        )
    # v5.9: 所有重试耗尽时返回带有 _degraded 标记的结构化降级值
    if fallback is None:
        return _build_exhausted_return(step_name)
    return fallback
