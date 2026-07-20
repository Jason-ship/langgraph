"""Features API — 特性开关端点。

Migrated from DeerFlow app/gateway/routers/features.py.

前端通过该端点获知当前启用了哪些功能。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["features"])


@router.get("/features")
async def get_features():
    """获取当前启用的特性列表。"""
    from novelfactory.config.settings import settings

    return {
        "channels": bool(settings.LARK_APP_ID),
        "memory": True,
        "feedback": True,
        "suggestions": True,
        "input_polish": True,
        "console": True,
        "trace": True,
    }


__all__ = ["router"]