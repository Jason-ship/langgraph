"""Scheduler — 定时任务调度器（租约式轮询调度）。

Migrated from DeerFlow app/scheduler/service.py.

基于 asyncio 轮询的定时任务调度器，使用租约机制防止多实例冲突。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────────────

_LEASE_SECONDS = 30
_LEASE_GRACE_SECONDS = 10
_POLL_INTERVAL_SECONDS = 5.0
_DEFAULT_MAX_CONCURRENT = 5


class ScheduledTask:
    """定时任务数据类。"""

    def __init__(
        self,
        task_id: str,
        cron_expr: str | None = None,
        *,
        thread_id: str = "",
        payload: dict[str, Any] | None = None,
        enabled: bool = True,
        max_concurrent: int = 1,
    ):
        self.task_id = task_id
        self.cron_expr = cron_expr
        self.thread_id = thread_id
        self.payload = payload or {}
        self.enabled = enabled
        self.max_concurrent = max_concurrent
        self.last_run_at: str | None = None
        self.next_run_at: str | None = None
        self.lease_owner: str | None = None
        self.lease_expires_at: str | None = None


class SchedulerService:
    """定时任务调度器。

    使用 asyncio 轮询 + 租约机制，支持多实例共存。
    """

    def __init__(self, max_concurrent: int = _DEFAULT_MAX_CONCURRENT):
        self._tasks: dict[str, ScheduledTask] = {}
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._max_concurrent = max_concurrent
        self._worker_id = f"{uuid.uuid4().hex[:8]}"
        self._running = False
        self._poll_task: asyncio.Task | None = None

    def register(self, task: ScheduledTask) -> None:
        """注册定时任务。"""
        self._tasks[task.task_id] = task
        logger.info("[scheduler] Registered task: %s", task.task_id)

    def unregister(self, task_id: str) -> None:
        """注销定时任务。"""
        self._tasks.pop(task_id, None)
        logger.info("[scheduler] Unregistered task: %s", task_id)

    async def start(self) -> None:
        """启动调度器。"""
        if self._running:
            return
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("[scheduler] Started (worker=%s)", self._worker_id)

    async def stop(self) -> None:
        """停止调度器。"""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        # 取消所有运行中的任务
        for task_id, run_task in list(self._running_tasks.items()):
            run_task.cancel()
        logger.info("[scheduler] Stopped")

    async def _poll_loop(self) -> None:
        """轮询循环：检查到期的任务并执行。"""
        while self._running:
            try:
                await self._check_and_dispatch()
            except Exception:
                logger.exception("[scheduler] Poll error")
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    async def _check_and_dispatch(self) -> None:
        """检查并分发到期的任务。"""
        if len(self._running_tasks) >= self._max_concurrent:
            return

        now = datetime.now(UTC)
        for task in list(self._tasks.values()):
            if not task.enabled:
                continue
            if task.task_id in self._running_tasks:
                continue

            # 检查租约是否过期
            if task.lease_expires_at:
                expires = datetime.fromisoformat(task.lease_expires_at)
                if expires > now:
                    continue  # 其他实例正在处理

            # 尝试获取租约
            task.lease_owner = self._worker_id
            task.lease_expires_at = (now + timedelta(seconds=_LEASE_SECONDS)).isoformat()
            task.last_run_at = now.isoformat()

            # 启动任务
            run_task = asyncio.create_task(self._execute_task(task))
            self._running_tasks[task.task_id] = run_task
            run_task.add_done_callback(lambda t, tid=task.task_id: self._running_tasks.pop(tid, None))

            if len(self._running_tasks) >= self._max_concurrent:
                break

    async def _execute_task(self, task: ScheduledTask) -> None:
        """执行定时任务。

        子类应重写此方法以实现具体业务逻辑。
        """
        logger.info("[scheduler] Executing task: %s", task.task_id)
        try:
            # 默认实现：打印日志
            logger.info("[scheduler] Task %s completed", task.task_id)
        except Exception as exc:
            logger.error("[scheduler] Task %s failed: %s", task.task_id, exc)
        finally:
            task.lease_owner = None
            task.lease_expires_at = None


__all__ = [
    "ScheduledTask",
    "SchedulerService",
]