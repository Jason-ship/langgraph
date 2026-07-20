"""Persistence 持久层 — 运行记录、反馈、线程元数据。

Migrated from DeerFlow persistence/run/, persistence/feedback/, persistence/thread_meta/.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


# ── 运行记录存储 ──────────────────────────────────────────────────────────


class RunStore:
    """运行记录存储（内存实现）。

    记录 Agent 运行的状态、Token 用量、模型信息等。
    """

    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}
        self._runs_by_thread: dict[str, list[str]] = {}

    def put(self, run_id: str, data: dict[str, Any]) -> None:
        """插入或更新运行记录。"""
        self._runs[run_id] = {**self._runs.get(run_id, {}), **data}
        thread_id = data.get("thread_id", "")
        if thread_id:
            if thread_id not in self._runs_by_thread:
                self._runs_by_thread[thread_id] = []
            if run_id not in self._runs_by_thread[thread_id]:
                self._runs_by_thread[thread_id].append(run_id)

    def get(self, run_id: str) -> dict[str, Any] | None:
        return self._runs.get(run_id)

    def list_by_thread(self, thread_id: str) -> list[dict[str, Any]]:
        run_ids = self._runs_by_thread.get(thread_id, [])
        return [self._runs[rid] for rid in run_ids if rid in self._runs]

    def update_status(self, run_id: str, status: str) -> None:
        if run_id in self._runs:
            self._runs[run_id]["status"] = status
            self._runs[run_id]["updated_at"] = datetime.now(UTC).isoformat()

    def delete(self, run_id: str) -> None:
        self._runs.pop(run_id, None)
        for thread_id, run_ids in list(self._runs_by_thread.items()):
            if run_id in run_ids:
                run_ids.remove(run_id)

    def aggregate_tokens_by_thread(self, thread_id: str) -> dict[str, int]:
        total = 0
        for run in self.list_by_thread(thread_id):
            total += run.get("total_tokens", 0) or 0
        return {"total_tokens": total}

    def list_pending(self) -> list[dict[str, Any]]:
        return [r for r in self._runs.values() if r.get("status") in ("pending", "running")]


# ── 反馈存储 ──────────────────────────────────────────────────────────────


class FeedbackStore:
    """用户反馈存储（内存实现）。"""

    def __init__(self) -> None:
        self._feedback: list[dict[str, Any]] = []

    def create(self, thread_id: str, run_id: str, rating: int, comment: str | None = None, user_id: str = "") -> dict[str, Any]:
        feedback = {
            "feedback_id": str(uuid.uuid4()),
            "thread_id": thread_id,
            "run_id": run_id,
            "user_id": user_id,
            "rating": rating,
            "comment": comment,
            "created_at": datetime.now(UTC).isoformat(),
        }
        self._feedback.append(feedback)
        return feedback

    def list_by_run(self, run_id: str) -> list[dict[str, Any]]:
        return [fb for fb in self._feedback if fb["run_id"] == run_id]

    def list_by_thread(self, thread_id: str) -> list[dict[str, Any]]:
        return [fb for fb in self._feedback if fb["thread_id"] == thread_id]

    def aggregate_by_run(self, run_id: str) -> dict[str, int]:
        items = [fb for fb in self._feedback if fb["run_id"] == run_id]
        return {"total": len(items), "positive": sum(1 for fb in items if fb["rating"] > 0), "negative": sum(1 for fb in items if fb["rating"] < 0)}


# ── 线程元数据存储 ────────────────────────────────────────────────────────


class ThreadMetaStore:
    """线程元数据存储（内存实现）。"""

    def __init__(self) -> None:
        self._threads: dict[str, dict[str, Any]] = {}

    def create(self, thread_id: str, data: dict[str, Any]) -> dict[str, Any]:
        meta = {
            "thread_id": thread_id,
            "display_name": data.get("display_name", ""),
            "status": data.get("status", "idle"),
            "metadata_json": json.dumps(data.get("metadata", {})),
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self._threads[thread_id] = meta
        return meta

    def get(self, thread_id: str) -> dict[str, Any] | None:
        return self._threads.get(thread_id)

    def check_access(self, thread_id: str, user_id: str) -> bool:
        thread = self._threads.get(thread_id)
        if thread is None:
            return False
        return True  # 简化版：所有用户都可访问

    def update_display_name(self, thread_id: str, name: str) -> None:
        if thread_id in self._threads:
            self._threads[thread_id]["display_name"] = name
            self._threads[thread_id]["updated_at"] = datetime.now(UTC).isoformat()

    def update_status(self, thread_id: str, status: str) -> None:
        if thread_id in self._threads:
            self._threads[thread_id]["status"] = status
            self._threads[thread_id]["updated_at"] = datetime.now(UTC).isoformat()

    def delete(self, thread_id: str) -> None:
        self._threads.pop(thread_id, None)


# ── 全局单例 ──────────────────────────────────────────────────────────────

_run_store = RunStore()
_feedback_store = FeedbackStore()
_thread_meta_store = ThreadMetaStore()


def get_run_store() -> RunStore:
    return _run_store


def get_feedback_store() -> FeedbackStore:
    return _feedback_store


def get_thread_meta_store() -> ThreadMetaStore:
    return _thread_meta_store


__all__ = [
    "RunStore",
    "FeedbackStore",
    "ThreadMetaStore",
    "get_run_store",
    "get_feedback_store",
    "get_thread_meta_store",
]