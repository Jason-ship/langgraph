"""Subagent 执行引擎 — 子 Agent 独立执行。

Migrated from DeerFlow subagents/executor.py + registry.py + config.py.

在独立线程中执行子 Agent，支持超时、取消、Token 收集。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SubagentStatus(Enum):
    """子 Agent 执行状态。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

    @property
    def is_terminal(self) -> bool:
        return self in {
            SubagentStatus.COMPLETED,
            SubagentStatus.FAILED,
            SubagentStatus.CANCELLED,
            SubagentStatus.TIMED_OUT,
        }


@dataclass
class SubagentResult:
    """子 Agent 执行结果。"""

    task_id: str
    status: SubagentStatus
    result: str = ""
    error: str | None = None
    stop_reason: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SubagentConfig:
    """子 Agent 配置。"""

    name: str = ""
    description: str = ""
    system_prompt: str = ""
    allowed_tools: list[str] | None = None
    model: str = ""
    max_turns: int = 20
    timeout_seconds: int = 120


class SubagentExecutor:
    """子 Agent 执行器。

    在独立线程中执行子 Agent 任务，支持超时和取消。
    """

    def __init__(self, max_concurrency: int = 5):
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._running: dict[str, asyncio.Task] = {}

    async def execute(
        self,
        config: SubagentConfig,
        task_input: str,
        *,
        timeout: int | None = None,
    ) -> SubagentResult:
        """执行子 Agent 任务。

        Args:
            config: 子 Agent 配置。
            task_input: 任务输入文本。
            timeout: 超时秒数（默认使用 config.timeout_seconds）。

        Returns:
            子 Agent 执行结果。
        """
        task_id = str(uuid.uuid4())
        result = SubagentResult(task_id=task_id, status=SubagentStatus.PENDING)
        timeout = timeout or config.timeout_seconds

        async with self._semaphore:
            result.status = SubagentStatus.RUNNING
            task = asyncio.create_task(self._run_agent(config, task_input, result))
            self._running[task_id] = task

            try:
                await asyncio.wait_for(task, timeout=timeout)
            except asyncio.TimeoutError:
                result.status = SubagentStatus.TIMED_OUT
                result.error = f"Timed out after {timeout}s"
                task.cancel()
            except asyncio.CancelledError:
                result.status = SubagentStatus.CANCELLED
                result.error = "Cancelled"
            except Exception as exc:
                result.status = SubagentStatus.FAILED
                result.error = str(exc)
            finally:
                self._running.pop(task_id, None)
                result.completed_at = datetime.now(UTC).isoformat()

        return result

    async def _run_agent(self, config: SubagentConfig, task_input: str, result: SubagentResult) -> None:
        """运行子 Agent。

        子类应重写此方法以实现具体的 Agent 执行逻辑。
        """
        logger.info("[subagent] Running: %s (input=%s...)", config.name, task_input[:100])
        result.result = f"Task completed by {config.name}"
        result.status = SubagentStatus.COMPLETED

    def cancel(self, task_id: str) -> None:
        """取消正在运行的任务。"""
        task = self._running.get(task_id)
        if task:
            task.cancel()


__all__ = [
    "SubagentStatus",
    "SubagentResult",
    "SubagentConfig",
    "SubagentExecutor",
]