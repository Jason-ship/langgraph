"""Feedback API — 反馈收集端点。

Migrated from DeerFlow app/gateway/routers/feedback.py.

允许用户提交 thumbs-up/down 反馈，可选关联到特定消息。
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from novelfactory.store.feedback_store import FeedbackStore, get_feedback_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/threads", tags=["feedback"])


class FeedbackCreateRequest(BaseModel):
    rating: int = Field(..., description="评分: +1 (正面) 或 -1 (负面)")
    comment: str | None = Field(default=None, description="可选反馈文本")
    message_id: str | None = Field(default=None, description="可选: 关联到特定消息")


class FeedbackResponse(BaseModel):
    feedback_id: str
    thread_id: str
    run_id: str
    rating: int
    comment: str | None = None
    created_at: str = ""


class FeedbackStatsResponse(BaseModel):
    thread_id: str
    run_id: str
    total: int = 0
    positive: int = 0
    negative: int = 0


# ── 存储（v8.4 落库修复：PostgreSQL 持久化，无 DB 时回退内存） ──

_feedback_store: FeedbackStore | None = None


def _get_store() -> FeedbackStore:
    """获取共享 FeedbackStore。"""
    global _feedback_store
    if _feedback_store is None:
        _feedback_store = get_feedback_store()
    return _feedback_store


@router.post("/{thread_id}/runs/{run_id}/feedback", response_model=FeedbackResponse)
async def create_feedback(thread_id: str, run_id: str, body: FeedbackCreateRequest):
    """提交反馈。

    对指定线程的某次运行提交 thumbs-up/down 反馈。
    """
    if body.rating not in (1, -1):
        raise HTTPException(status_code=422, detail="Rating must be +1 or -1")

    feedback = {
        "feedback_id": str(uuid.uuid4()),
        "thread_id": thread_id,
        "run_id": run_id,
        "rating": body.rating,
        "comment": body.comment,
        "message_id": body.message_id,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _get_store().create(feedback)

    logger.info("[feedback] Created: thread=%s run=%s rating=%+d", thread_id, run_id, body.rating)
    return FeedbackResponse(**feedback)


@router.get("/{thread_id}/runs/{run_id}/feedback", response_model=list[FeedbackResponse])
async def list_feedback(thread_id: str, run_id: str):
    """获取反馈列表。"""
    return [
        FeedbackResponse(**fb)
        for fb in _get_store().list_by_run(thread_id, run_id)
    ]


@router.get("/{thread_id}/runs/{run_id}/feedback/stats", response_model=FeedbackStatsResponse)
async def feedback_stats(thread_id: str, run_id: str):
    """获取反馈统计。"""
    items = _get_store().list_by_run(thread_id, run_id)
    return FeedbackStatsResponse(
        thread_id=thread_id,
        run_id=run_id,
        total=len(items),
        positive=sum(1 for fb in items if fb["rating"] > 0),
        negative=sum(1 for fb in items if fb["rating"] < 0),
    )


@router.delete("/{thread_id}/runs/{run_id}/feedback/{feedback_id}")
async def delete_feedback(thread_id: str, run_id: str, feedback_id: str):
    """删除反馈。"""
    if not _get_store().delete(feedback_id, thread_id, run_id):
        raise HTTPException(status_code=404, detail="Feedback not found")
    return {"status": "deleted"}
