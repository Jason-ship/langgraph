"""Long-Term Memory Nodes (BaseStore) for cross-session project persistence."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.runtime import Runtime

from novelfactory.state.novel_context import NovelContext
from novelfactory.state.novel_state import NovelFactoryState

logger = logging.getLogger(__name__)


def load_longterm_memory(
    state: NovelFactoryState, *, runtime: Runtime[NovelContext], store: Any = None
) -> dict:
    """Load cross-session project memory from BaseStore."""
    if store is None:
        logger.info("[store] No store configured, skipping load")
        return {}
    project_id = runtime.context.get("project_id") or state.get(
        "project_name", "default"
    )
    namespace = ("novelfactory", project_id)

    try:
        item = store.get(namespace, "project_meta")
        if item:
            logger.info("[store] Loaded long-term memory for project_id=%s", project_id)
            return {"loaded_memory": item.value or {}}
    except Exception as e:
        logger.warning("[store] Failed to load memory: %s", e)

    return {"loaded_memory": {}}


def save_longterm_memory(
    state: NovelFactoryState, *, runtime: Runtime[NovelContext], store: Any = None
) -> dict:
    """Save project metadata to BaseStore on project completion."""
    updates: dict = {}

    # v6.1 P3-1: 生成 WallTimeTracker 全链路耗时报告
    try:
        from novelfactory.utils.wall_time_tracker import create_tracker_from_state

        tracker = create_tracker_from_state(state)
        if tracker._records:
            report = tracker.report()
            logger.info("[store] WallTime Report:\n%s", report)
            import json as _json

            updates["wall_time_data"] = _json.dumps(tracker.to_dict())
    except Exception as e:
        logger.debug("[store] WallTime report generation failed: %s", e)

    if store is None:
        logger.info("[store] No store configured, skipping save")
        return updates
    project_id = runtime.context.get("project_id") or state.get(
        "project_name", "default"
    )
    namespace = ("novelfactory", project_id)

    completed = state.get("completed_chapters", [])
    meta = {
        "genre": state.get("genre", ""),
        "target_chapters": state.get("target_chapters", 0),
        "completed_count": len(completed),
        "word_count": state.get("word_count", 0),
        "project_name": state.get("project_name", ""),
    }

    try:
        store.put(namespace, "project_meta", meta)
        logger.info("[store] Saved project meta for %s: %s", project_id, meta)

        thread_id = state.get("thread_id", "")
        try:
            from novelfactory.integrations.feishu.notify import (
                send_progress_notification,
            )

            send_progress_notification(
                thread_id=thread_id,
                chapter=meta.get("completed_count", 0),
                total=meta.get("target_chapters", 0),
            )
        except Exception as e:
            logger.warning("[store] Feishu completion notification failed: %s", e)
    except Exception as e:
        logger.warning("[store] Failed to save memory: %s", e)

    # ── 检查点终态清理（v5.6: 小说完成后清理 PG 检查点） ──
    if state.get("current_phase") == "done":
        try:
            import asyncio

            from novelfactory.graph.checkpointer import (
                get_checkpointer_instance,
                maybe_cleanup_checkpoints,
            )

            cp = get_checkpointer_instance()
            if cp is not None:
                thread_id = runtime.context.get("thread_id") or state.get(
                    "thread_id", ""
                )
                cfg = {"configurable": {"thread_id": thread_id}} if thread_id else None
                asyncio.create_task(
                    maybe_cleanup_checkpoints(state, config=cfg, checkpointer=cp)
                )
        except Exception as e:
            logger.warning("[checkpoint] 清理检查点时出错: %s", e)

    return updates
