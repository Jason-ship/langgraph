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


# ── 内存存储（简化版） ──

_feedback_store: list[dict] = []


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
    _feedback_store.append(feedback)

    logger.info("[feedback] Created: thread=%s run=%s rating=%+d", thread_id, run_id, body.rating)
    return FeedbackResponse(**feedback)


@router.get("/{thread_id}/runs/{run_id}/feedback", response_model=list[FeedbackResponse])
async def list_feedback(thread_id: str, run_id: str):
    """获取反馈列表。"""
    results = [
        FeedbackResponse(**fb)
        for fb in _feedback_store
        if fb["thread_id"] == thread_id and fb["run_id"] == run_id
    ]
    return results


@router.get("/{thread_id}/runs/{run_id}/feedback/stats", response_model=FeedbackStatsResponse)
async def feedback_stats(thread_id: str, run_id: str):
    """获取反馈统计。"""
    items = [fb for fb in _feedback_store if fb["thread_id"] == thread_id and fb["run_id"] == run_id]
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
    global _feedback_store
    before = len(_feedback_store)
    _feedback_store = [
        fb
        for fb in _feedback_store
        if not (fb["feedback_id"] == feedback_id and fb["thread_id"] == thread_id and fb["run_id"] == run_id)
    ]
    if len(_feedback_store) == before:
        raise HTTPException(status_code=404, detail="Feedback not found")
    return {"status": "deleted"}