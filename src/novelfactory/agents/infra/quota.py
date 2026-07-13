"""Per-call token budget enforcement using accumulated usage tracking.

Replaces the dead DeepSeek billing API client (integrations/minimax/quota.py,
removed 2026-07-01) with a simple check against the token accumulator in
agents/infra/usage.py.

TODO: Wire up a real billing API backend (DeepSeek, ARK, etc.) when one
      becomes available. The current implementation only enforces a
      configurable per-run token budget via QUOTA_TOKEN_BUDGET.
"""

from __future__ import annotations

import time as _time_module
from typing import Any

from novelfactory.agents.infra.logger import get_logger

# In-process cache: last evaluation timestamp (for interval-based rate limiting)
_quota_cache: dict = {"last_fetch": 0.0}


# ── Config helpers (deferred import to avoid circular deps) ─────────────────────


def _quota_check_interval() -> float:
    try:
        from novelfactory.config.quota import quota_settings

        return quota_settings.QUOTA_CHECK_INTERVAL_SECONDS
    except (ImportError, AttributeError):
        return 60.0


def _quota_check_before_call() -> bool:
    """Return True if per-call quota check is enabled (QUOTA_CHECK_BEFORE_CALL)."""
    try:
        from novelfactory.config.quota import quota_settings

        return quota_settings.QUOTA_CHECK_BEFORE_CALL
    except (ImportError, AttributeError):
        return False


def _quota_warn_threshold() -> float:
    """Return the warning threshold percentage (QUOTA_WARN_THRESHOLD_PCT)."""
    try:
        from novelfactory.config.quota import quota_settings

        return float(quota_settings.QUOTA_WARN_THRESHOLD_PCT)
    except (ImportError, AttributeError):
        return 20.0


def _quota_block_threshold() -> float:
    """Return the block threshold percentage (QUOTA_BLOCK_THRESHOLD_PCT)."""
    try:
        from novelfactory.config.quota import quota_settings

        return float(quota_settings.QUOTA_BLOCK_THRESHOLD_PCT)
    except (ImportError, AttributeError):
        return 5.0


def _format_reset_time(seconds: int) -> str:
    """将秒数格式化为人类可读的时间字符串。

    Args:
        seconds: 剩余秒数

    Returns:
        "5m", "1h 30m", "24h 0m", "unknown" 等格式
    """
    if seconds <= 0:
        return "unknown"
    minutes = seconds // 60
    hours = minutes // 60
    if hours > 0:
        remaining_minutes = minutes % 60
        return f"{hours}h {remaining_minutes}m"
    return f"{minutes}m"


# ── Core check: called by retry.py / async_retry.py ─────────────────────────────


def _check_quota_safe() -> tuple[bool, str]:
    """Check accumulated token usage against the configured budget.

    Returns (blocked, reason):
      - blocked=False, reason="" — within budget or no budget configured
      - blocked=True, reason="..." — budget exhausted, should block calls

    Respects QUOTA_CHECK_INTERVAL_SECONDS to avoid thrashing on every call.
    When QUOTA_TOKEN_BUDGET is 0 (disabled), returns immediately as non-blocking.
    """
    _log = get_logger("novelfactory.llm")  # noqa: F841

    # ── Rate-limit checks ──
    now = _time_module.time()
    last = _quota_cache.get("last_fetch", 0.0)
    interval = _quota_check_interval()
    if now - last < interval:
        return False, ""

    _quota_cache["last_fetch"] = now

    # ── Resolve budget ──
    try:
        from novelfactory.config.quota import quota_settings

        budget = quota_settings.QUOTA_TOKEN_BUDGET
        warn_pct = quota_settings.QUOTA_WARN_THRESHOLD_PCT
        block_pct = quota_settings.QUOTA_BLOCK_THRESHOLD_PCT
    except (ImportError, AttributeError):
        return False, ""

    if budget <= 0:
        return False, ""

    # ── Read accumulated usage ──
    from novelfactory.agents.infra.usage import read_usage_tracking

    try:
        usage = read_usage_tracking()
        total_tokens: int = int(usage.get("total_tokens", 0))
        estimated_cost: float = float(usage.get("estimated_cost_cny", 0.0))
    except Exception:
        return False, ""

    if total_tokens <= 0:
        return False, ""

    remaining = max(0, budget - total_tokens)
    remaining_pct = remaining / budget * 100

    # ── Evaluate against thresholds ──

    if remaining_pct <= block_pct:
        return True, (
            f"token budget critical: {total_tokens:,}/{budget:,} tokens "
            f"({remaining_pct:.1f}% remaining, estimated cost: ¥{estimated_cost:.2f})"
        )

    if remaining_pct <= warn_pct:
        return False, (
            f"warn:budget_low:{total_tokens:,}/{budget:,} tokens "
            f"({remaining_pct:.1f}% remaining)"
        )

    return False, ""


# ── Public API: called by graph nodes and state introspection ───────────────────


def refresh_quota() -> dict[str, Any] | None:
    """Return current token budget status from the usage accumulator.

    Called by graph/nodes/quota.py's refresh_quota_node to populate
    state.quota_info at project start and between writing cycles.

    TODO: Replace with real billing API when available.
    """
    from novelfactory.agents.infra.usage import read_usage_tracking

    try:
        usage = read_usage_tracking()
        try:
            from novelfactory.config.quota import quota_settings

            budget = quota_settings.QUOTA_TOKEN_BUDGET
        except (ImportError, AttributeError):
            budget = 0

        total = int(usage.get("total_tokens", 0))
        return {
            "total_tokens": total,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "estimated_cost_cny": float(usage.get("estimated_cost_cny", 0.0)),
            "budget": budget,
            "remaining": max(0, budget - total) if budget > 0 else 0,
            "calls": len(usage.get("calls", [])),
            "backend": "usage_accumulator",
        }
    except Exception:
        return {
            "total_tokens": 0,
            "budget": 0,
            "remaining": 0,
            "backend": "usage_accumulator",
        }


def get_quota_status() -> dict[str, Any]:
    """Return current quota status (no external API call).

    Used for administrative introspection via agents/infra/__init__.py.
    """
    return refresh_quota() or {"cached": True, "available": False}
