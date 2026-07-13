"""ChapterStateTracker persistence via LangGraph Store (BaseStore API).

桥接 ChapterStateTracker ↔ LangGraph BaseStore，实现跨会话持久化。
写前加载 tracker 状态注入 writer prompt，写后保存 tracker 状态供下一章使用。

Usage:
    from novelfactory.store.chapter_state_store import (
        save_tracker_to_store,
        load_tracker_from_store,
    )

    # 写后保存
    await save_tracker_to_store(store, project_name, tracker)

    # 写前加载
    tracker = await load_tracker_from_store(store, project_name, world, chars, outlines)
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from novelfactory.state.chapter_state import ChapterStateTracker

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore

logger = logging.getLogger(__name__)

# ── Store namespace constants ───────────────────────────────────────────────────
_NAMESPACE = ("chapter_state",)
_TRACKER_KEY = "tracker"

# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


async def save_tracker_to_store(
    store: BaseStore,
    project_name: str,
    tracker: ChapterStateTracker,
) -> bool:
    """Persist ChapterStateTracker to LangGraph BaseStore.

    Args:
        store: LangGraph BaseStore instance (AsyncPostgresStore or InMemoryStore).
        project_name: Project identifier (used as namespace segment).
        tracker: ChapterStateTracker instance to persist.

    Returns:
        True on success, False on failure.
    """
    try:
        namespace = _NAMESPACE + (project_name,)
        data = tracker.to_dict()
        # Store as JSON-serializable dict via the store's put API
        await store.aput(
            namespace, _TRACKER_KEY, {"data": json.dumps(data, ensure_ascii=False)}
        )
        logger.debug(
            "ChapterStateTracker saved to store: project=%s, characters=%d, threads=%d",
            project_name,
            len(data.get("characters", {})),
            len(data.get("unresolved_threads", [])),
        )
        return True
    except Exception as e:
        logger.warning("Failed to save ChapterStateTracker to store: %s", e)
        return False


async def load_tracker_from_store(
    store: BaseStore,
    project_name: str,
    world_setting: str = "",
    character_setting: str = "",
    chapter_outlines: str = "",
) -> ChapterStateTracker | None:
    """Load ChapterStateTracker from LangGraph BaseStore.

    Args:
        store: LangGraph BaseStore instance.
        project_name: Project identifier.
        world_setting: 世界观设定（新建 tracker 时使用）。
        character_setting: 角色设定（新建 tracker 时使用）。
        chapter_outlines: 章节大纲（新建 tracker 时使用）。

    Returns:
        ChapterStateTracker if found/created, None on critical failure.
    """
    try:
        namespace = _NAMESPACE + (project_name,)
        item = await store.aget(namespace, _TRACKER_KEY)

        if item and item.get("value"):
            value = item["value"]
            # Value may be a dict with "data" key (as stored) or direct dict
            raw = value.get("data", value) if isinstance(value, dict) else value
            if isinstance(raw, str):
                raw = json.loads(raw)
            if isinstance(raw, dict) and raw.get("characters"):
                tracker = ChapterStateTracker.from_dict(raw)
                # Restore setup context (not persisted in store to save space)
                tracker.world_setting = world_setting or tracker.world_setting
                tracker.character_setting = (
                    character_setting or tracker.character_setting
                )
                tracker.chapter_outlines = chapter_outlines or tracker.chapter_outlines
                logger.debug(
                    "ChapterStateTracker loaded from store: project=%s, chapter=%d, characters=%d",
                    project_name,
                    raw.get("last_chapter_number", 0),
                    len(raw.get("characters", {})),
                )
                return tracker
        logger.debug(
            "No ChapterStateTracker found in store for project=%s, creating new",
            project_name,
        )
    except Exception as e:
        logger.warning("Failed to load ChapterStateTracker from store: %s", e)

    # Create new tracker with setup context
    return ChapterStateTracker(
        world_setting=world_setting,
        character_setting=character_setting,
        chapter_outlines=chapter_outlines,
    )
