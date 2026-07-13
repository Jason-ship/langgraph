"""异步 LLM 调用包装器 — timeout + error-type-aware retry + usage auditing.

异步版本对应 ``retry.py`` 的 ``llm_call_with_retry``。
使用 ``asyncio.wait_for`` 替代 threading 超时，``asyncio.sleep`` 替代 ``time.sleep``。
保持相同的重试策略、熔断器检查、配额检查、用量审计。

Shared helpers live in ``_retry_common.py``.
"""

from __future__ import annotations

import asyncio
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
from novelfactory.agents.infra.timeout import LLMTimeoutError
from novelfactory.agents.infra.usage import (
    _extract_usage_from_result,
    _record_usage,
)
from novelfactory.config.constants import DEFAULT_TIMEOUT

# ── Retry Configuration ───────────────────────────────────────────────────────

_DEFAULT_TIMEOUT = DEFAULT_TIMEOUT


# ── Main async retry wrapper ───────────────────────────────────────────────────


async def async_llm_call_with_retry(
    func: Any,
    *args: Any,
    step_name: str = "llm_call",
    retry_policy: str = "default",
    timeout_seconds: float = _DEFAULT_TIMEOUT,
    fallback: Any = None,
    cache_prompt: str | None = None,
    cache_model: str = "deepseek-v4-flash",
    cache_temperature: float = 0.0,
    **kwargs: Any,
) -> Any:
    """异步 LLM 调用包装器 — timeout + error-type-aware retry + optional cache.

    与 ``llm_call_with_retry`` 保持相同的重试策略:
      - 400/401/403: 永不重试
      - 429: 立即重试
      - 500/502/503/504: 指数退避
      - TimeoutError/OSError: 指数退避

    异步差异:
      - 使用 ``asyncio.wait_for`` 替代 threading ``with_timeout``
      - 使用 ``asyncio.sleep`` 替代 ``time.sleep``
      - ``func`` 必须是异步函数 (支持 await)

    v5.4 缓存:
      - ``cache_prompt``: 如果提供，LLM 调用前查缓存，成功后写入缓存
      - ``cache_model`` + ``cache_temperature``: 缓存键前缀
    """
    _call_start = time.time()
    logger = get_logger("novelfactory.llm")
    policy = _resolve_retry_policy(retry_policy)
    max_retries = policy.max_attempts

    # ── Cache lookup (v5.4) ──
    if cache_prompt:
        try:
            from novelfactory.agents.infra.llm_cache import get_llm_cache

            cache = get_llm_cache()
            if cache.available:
                cached = await cache.get(cache_model, cache_temperature, cache_prompt)
                if cached:
                    logger.info(
                        "[%s] Cache HIT: model=%s t=%.2f — skipping LLM call",
                        step_name,
                        cache_model,
                        cache_temperature,
                    )
                    import json as _json

                    try:
                        return _json.loads(cached)
                    except Exception:
                        return cached
        except Exception:
            pass  # 缓存失败不影响主流程

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
            # asyncio.wait_for 替代 with_timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=timeout_seconds,
            )

            if result is None:
                _msg = f"{step_name} timed out after {timeout_seconds}s"
                raise LLMTimeoutError(_msg)

            # 截断检测 (finish_reason=length)
            truncated = _check_truncation(result)
            if truncated:
                logger.warning(
                    "[%s] attempt %d hit max_tokens ceiling (finish_reason=length). "
                    "Retrying.",
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

            # v5.4: 写入 LLM 响应缓存
            if cache_prompt and isinstance(result, dict):
                try:
                    from novelfactory.agents.infra.llm_cache import get_llm_cache

                    cache = get_llm_cache()
                    if cache.available:
                        import json as _json

                        await cache.set(
                            cache_model,
                            cache_temperature,
                            cache_prompt,
                            _json.dumps(result, ensure_ascii=False),
                        )
                except Exception:
                    pass

            return result

        # ── asyncio 原生超时 ──
        except asyncio.TimeoutError:
            last_err = LLMTimeoutError(
                f"{step_name} async timeout after {timeout_seconds}s"
            )
            _record_provider_failures()
            delay = compute_backoff_delay(
                attempt,
                policy.initial_interval,
                getattr(policy, "max_interval", 60.0),
            )
            logger.warning(
                "[%s] attempt %d/%d — asyncio.TimeoutError — retrying in %ds",
                step_name,
                attempt,
                max_retries,
                delay,
            )
            if attempt < max_retries:
                await asyncio.sleep(delay)

        # ── 错误分类与重试 ──
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
                    "[%s] attempt %d/%d — rate limited (HTTP %s) "
                    "— retrying immediately",
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
                await asyncio.sleep(delay)

    if last_err:
        logger.error(
            "[%s] failed after %d attempts: %s",
            step_name,
            max_retries,
            last_err,
        )
    # v5.9: 所有重试耗尽时返回带有 _degraded 标记的结构化降级值
    if fallback is None:
        return _build_exhausted_return(step_name)
    return fallback
