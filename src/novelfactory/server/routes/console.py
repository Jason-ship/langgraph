"""Console API — 跨线程可观测性端点。

Migrated from DeerFlow app/gateway/routers/console.py.

提供运行统计、Token 用量、线程状态等监控数据。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/console", tags=["console"])


class ConsoleStatsResponse(BaseModel):
    """运行统计概览。"""

    total_runs: int = Field(0, description="总运行次数")
    active_runs: int = Field(0, description="正在运行")
    failed_runs: int = Field(0, description="失败次数")
    total_threads: int = Field(0, description="总线程数")
    total_tokens: int = Field(0, description="Token 消耗总量")


class ConsoleRunItem(BaseModel):
    """运行记录项。"""

    run_id: str
    thread_id: str
    status: str
    model_name: str | None = None
    created_at: str | None = None
    duration_seconds: float | None = None
    total_tokens: int = 0
    error: str | None = None


class ConsoleRunsResponse(BaseModel):
    """运行记录列表。"""

    runs: list[ConsoleRunItem]


@router.get("/stats", response_model=ConsoleStatsResponse)
async def console_stats():
    """获取运行统计概览。

    返回总运行数、活跃数、失败数、线程数、Token 消耗。
    """
    from novelfactory.utils.wall_time_tracker import WallTimeTracker

    tracker = WallTimeTracker()
    summary = tracker.get_summary()

    return ConsoleStatsResponse(
        total_runs=summary.get("total_runs", 0),
        active_runs=summary.get("active_runs", 0),
        failed_runs=summary.get("failed_runs", 0),
        total_threads=summary.get("total_threads", 0),
        total_tokens=summary.get("total_tokens", 0),
    )


@router.get("/runs", response_model=ConsoleRunsResponse)
async def console_runs(
    limit: int = Query(default=20, ge=1, le=100, description="返回条数"),
    offset: int = Query(default=0, ge=0, description="偏移量"),
):
    """获取运行记录列表。

    Args:
        limit: 返回条数（1-100）。
        offset: 偏移量。
    """
    from novelfactory.utils.wall_time_tracker import WallTimeTracker

    tracker = WallTimeTracker()
    runs_data = tracker.get_recent_runs(limit=limit, offset=offset)

    return ConsoleRunsResponse(
        runs=[ConsoleRunItem(**run) for run in runs_data]
    )


@router.get("/runs/recent", response_model=ConsoleRunsResponse)
async def console_recent_runs(
    hours: int = Query(default=24, ge=1, le=168, description="最近 N 小时"),
):
    """获取最近 N 小时的运行记录。"""
    from novelfactory.utils.wall_time_tracker import WallTimeTracker

    tracker = WallTimeTracker()
    cutoff = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
    runs_data = tracker.get_recent_runs(cutoff=cutoff)

    return ConsoleRunsResponse(runs=[ConsoleRunItem(**run) for run in runs_data])