# ==============================================================================
# SDK: /runs/crons — 基于 PostgresStore 的持久化定时任务
#
# 使用 AsyncPostgresStore 存储 cron 定义（namespace="cron"），
# 配合 app.py lifespan 中的后台调度器执行到期任务。
# ==============================================================================

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter

from novelfactory.server.models import CronModel

logger = logging.getLogger(__name__)
router = APIRouter()

from novelfactory.server.deps import get_store  # noqa: E402

_CRON_NS = ("cron",)


# ── Schedule helpers ──────────────────────────────────────────────────────────


def _parse_cron(expr: str) -> tuple[int, int, int, int, int] | None:
    """Simple cron parser — returns (minute, hour, day, month, day_of_week)."""
    parts = expr.strip().split()
    if len(parts) != 5:
        return None
    try:
        return (
            int(parts[0]) if parts[0] != "*" else -1,
            int(parts[1]) if parts[1] != "*" else -1,
            int(parts[2]) if parts[2] != "*" else -1,
            int(parts[3]) if parts[3] != "*" else -1,
            int(parts[4]) if parts[4] != "*" else -1,
        )
    except ValueError:
        return None


def should_run_now(schedule: str, now: datetime) -> bool:
    """Check if a cron schedule matches the current time.

    Uses _parse_cron for basic 5-field cron expressions.
    Falls back to True (always run) for unsupported expressions
    (ranges, lists, step values) to maintain backward compatibility.

    Args:
        schedule: cron expression (e.g. "0 */6 * * *")
        now: current UTC datetime

    Returns:
        True if the cron should execute at this time
    """
    parsed = _parse_cron(schedule)
    if parsed is None:
        # Unsupported expression (ranges, lists, steps) — run every poll
        return True

    minute, hour, day, month, dow = parsed
    # -1 means wildcard (*)
    if minute != -1 and now.minute != minute:
        return False
    if hour != -1 and now.hour != hour:
        return False
    if day != -1 and now.day != day:
        return False
    if month != -1 and now.month != month:
        return False
    if dow != -1:
        # cron: 0=Sunday..6=Saturday; Python weekday: 0=Monday..6=Sunday
        py_weekday = (dow - 1) % 7
        if now.weekday() != py_weekday:
            return False
    return True


# ── Public API ────────────────────────────────────────────────────────────────


@router.post("/runs/crons", tags=["runs"], include_in_schema=False)
async def create_cron(cron: CronModel | None = None) -> dict:
    """Create a cron job (persisted to PostgresStore).

    安全校验：
      - 禁止 input 为空的 cron 创建（防止误创建后自动执行写作流程）
    """
    store = await get_store()
    entry = cron or CronModel()
    resolved = entry.model_dump()

    # ── [Safety] 空 input 校验 ──────────────────────────────────────────
    inp = resolved.get("input", {})
    if not inp or inp == {}:
        logger.warning(
            "[cron] 拒绝创建空 input 的 cron — cron=%s", resolved.get("cron_id", "")
        )
        return {
            "error": "cron input cannot be empty",
            "cron_id": resolved.get("cron_id", ""),
        }

    resolved["status"] = "active"
    resolved["created_at"] = datetime.now(timezone.utc).isoformat()
    resolved["updated_at"] = resolved["created_at"]
    await store.aput(_CRON_NS, resolved["cron_id"], resolved)
    logger.info(
        "[cron] Created cron %s (schedule=%s)",
        resolved["cron_id"],
        resolved["schedule"],
    )
    return {"cron_id": resolved["cron_id"]}


@router.post("/threads/{thread_id}/runs/crons", tags=["runs"], include_in_schema=False)
async def create_thread_cron(thread_id: str, cron: CronModel | None = None) -> dict:
    """Create a thread cron job."""
    store = await get_store()
    entry = cron or CronModel()
    resolved = entry.model_dump()
    resolved["thread_id"] = thread_id
    resolved["status"] = "active"
    resolved["created_at"] = datetime.now(timezone.utc).isoformat()
    resolved["updated_at"] = resolved["created_at"]
    await store.aput(_CRON_NS, resolved["cron_id"], resolved)
    return {"cron_id": resolved["cron_id"]}


@router.delete("/runs/crons/{cron_id}", tags=["runs"])
async def delete_cron(cron_id: str) -> dict:
    """Delete a cron job."""
    store = await get_store()
    await store.adelete(_CRON_NS, cron_id)
    return {"deleted": cron_id}


@router.post("/runs/crons/search", tags=["runs"])
async def search_crons() -> list:
    """Search cron jobs — lists all persisted cron definitions."""
    store = await get_store()
    try:
        items = await store.asearch(_CRON_NS, limit=1000)
        return [
            {
                "cron_id": item.key,
                "value": item.value,
            }
            for item in items
        ]
    except Exception as e:
        logger.warning("[cron] Failed to search crons: %s", e)
        return []
