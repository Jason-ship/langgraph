"""Quota refresh node for DeepSeek billing tracking."""

from __future__ import annotations

import logging

from novelfactory.state.novel_state import NovelFactoryState

logger = logging.getLogger(__name__)


def refresh_quota_node(state: NovelFactoryState) -> dict:
    """Fetch current DeepSeek billing quota and store it in state."""
    try:
        from novelfactory.agents.infra import refresh_quota as _refresh_quota

        snapshot = _refresh_quota()
        if snapshot:
            return {"quota_info": snapshot}
    except Exception as e:
        logger.warning("[quota_node] failed to refresh quota: %s", e)
    return {}
