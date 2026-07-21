"""Features API — 特性开关端点。

Migrated from DeerFlow app/gateway/routers/features.py.

前端通过该端点获知当前启用了哪些功能。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(tags=["features"])

FEATURES: dict[str, dict] = {
    "agents_api": {
        "enabled": True,
        "description": "Agent management API (CRUD)",
    },
    "skills_api": {
        "enabled": True,
        "description": "Skills system with Markdown skills",
    },
    "subtask_tracking": {
        "enabled": True,
        "description": "Sub-task tracking via SSE custom events",
    },
    "context_compaction": {
        "enabled": True,
        "description": "Thread context summarization and compaction",
    },
    "workspace_changes": {
        "enabled": False,
        "description": "Workspace file change tracking",
    },
    "voice_input": {
        "enabled": False,
        "description": "Voice dictation input",
    },
}


@router.get("/features")
async def get_features():
    """获取当前启用的特性列表。"""
    from novelfactory.config.settings import settings

    # Build dynamic response from FEATURES dict — each feature is wrapped
    # as {"enabled": bool} so the frontend can access data.agents_api.enabled.
    result: dict[str, dict[str, bool]] = {}
    for key, info in FEATURES.items():
        result[key] = {"enabled": info["enabled"]}

    # Add runtime-dependent features
    result["memory"] = {"enabled": True}
    result["channels"] = {"enabled": bool(settings.LARK_APP_ID)}
    result["mcp"] = {"enabled": False}
    result["artifacts"] = {"enabled": False}
    result["uploads"] = {"enabled": False}
    result["scheduled_tasks"] = {"enabled": True}
    result["input_polish"] = {"enabled": True}
    result["suggestions"] = {"enabled": True}
    result["console"] = {"enabled": True}
    result["feedback"] = {"enabled": True}
    result["trace"] = {"enabled": True}

    return result


__all__ = ["router"]