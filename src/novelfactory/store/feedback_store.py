"""FeedbackStore — 用户反馈持久化存储（v8.4 落库修复）。

背景: feedback 路由历史使用纯内存 list，进程重启即丢数据。
本模块提供 PostgreSQL 持久化（thread_feedback 表），
无数据库连接时自动回退内存存储，保持既有行为。
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class FeedbackStore:
    """用户反馈存储 — PostgreSQL 优先，内存回退。"""

    TABLE = "thread_feedback"

    def __init__(self, db_manager: Any | None = None) -> None:
        self._db = db_manager
        self._items: list[dict] = []
        if self._db is not None:
            try:
                self._ensure_schema()
            except Exception as exc:
                logger.warning("[FeedbackStore] schema init failed, fallback to memory: %s", exc)
                self._db = None

    @property
    def persistent(self) -> bool:
        """是否已接入 PostgreSQL。"""
        return self._db is not None

    def _ensure_schema(self) -> None:
        assert self._db is not None
        self._db.execute(
            f"CREATE TABLE IF NOT EXISTS {self.TABLE} ("
            "  feedback_id TEXT PRIMARY KEY,"
            "  thread_id TEXT NOT NULL,"
            "  run_id TEXT NOT NULL,"
            "  rating INTEGER NOT NULL,"
            "  comment TEXT,"
            "  message_id TEXT,"
            "  created_at TIMESTAMPTZ DEFAULT NOW()"
            ")"
        )
        self._db.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.TABLE}_thread_run "
            f"ON {self.TABLE} (thread_id, run_id)"
        )

    def create(self, feedback: dict) -> dict:
        """创建反馈。"""
        self._items.append(dict(feedback))
        if self._db is None:
            return feedback
        try:
            self._db.execute(
                f"INSERT INTO {self.TABLE} (feedback_id, thread_id, run_id, rating, comment, message_id) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    feedback["feedback_id"],
                    feedback["thread_id"],
                    feedback["run_id"],
                    feedback["rating"],
                    feedback.get("comment"),
                    feedback.get("message_id"),
                ),
            )
        except Exception as exc:
            logger.warning("[FeedbackStore] create db failed, kept in memory: %s", exc)
        return feedback

    def list_by_run(self, thread_id: str, run_id: str) -> list[dict]:
        """列出指定线程/运行的反馈。"""
        if self._db is not None:
            try:
                rows = self._db.fetchall(
                    f"SELECT feedback_id, thread_id, run_id, rating, comment, message_id, created_at "
                    f"FROM {self.TABLE} WHERE thread_id = %s AND run_id = %s ORDER BY created_at",
                    (thread_id, run_id),
                )
                return [self._row_to_dict(r) for r in rows]
            except Exception as exc:
                logger.warning("[FeedbackStore] list db failed, fallback: %s", exc)
        return [
            dict(fb)
            for fb in self._items
            if fb.get("thread_id") == thread_id and fb.get("run_id") == run_id
        ]

    def delete(self, feedback_id: str, thread_id: str, run_id: str) -> bool:
        """删除反馈，返回是否删除。"""
        before = len(self._items)
        self._items = [
            fb
            for fb in self._items
            if not (fb["feedback_id"] == feedback_id and fb["thread_id"] == thread_id and fb["run_id"] == run_id)
        ]
        deleted_mem = len(self._items) != before
        if self._db is not None:
            try:
                result = self._db.execute(
                    f"DELETE FROM {self.TABLE} WHERE feedback_id = %s AND thread_id = %s AND run_id = %s",
                    (feedback_id, thread_id, run_id),
                )
                deleted_db = result is not None and "DELETE" in str(result)
                return deleted_mem or deleted_db
            except Exception as exc:
                logger.warning("[FeedbackStore] delete db failed, fallback: %s", exc)
        return deleted_mem

    @staticmethod
    def _row_to_dict(row: tuple) -> dict:
        return {
            "feedback_id": str(row[0]),
            "thread_id": str(row[1]),
            "run_id": str(row[2]),
            "rating": int(row[3]),
            "comment": row[4],
            "message_id": row[5],
            "created_at": row[6].isoformat() if hasattr(row[6], "isoformat") else str(row[6]),
        }


# ── 模块级单例 ────────────────────────────────────────────────────────────────

_feedback_store_singleton: FeedbackStore | None = None
_feedback_store_lock = threading.Lock()


def _db_reachable(timeout: float = 1.0) -> bool:
    """快速 TCP 探测 PostgreSQL 可达性（避免 DatabaseManager 阻塞重试）。"""
    import socket

    try:
        from novelfactory.config.settings import settings as _st

        host = _st.DB_HOST or "localhost"
        port = int(_st.DB_PORT or 5432)
    except Exception:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def get_feedback_store() -> FeedbackStore:
    """获取共享 FeedbackStore 单例（懒加载 DatabaseManager）。"""
    global _feedback_store_singleton
    if _feedback_store_singleton is None:
        with _feedback_store_lock:
            if _feedback_store_singleton is None:
                db = None
                if _db_reachable():
                    try:
                        from novelfactory.config.database import DatabaseManager

                        db = DatabaseManager.get_instance()
                    except Exception as exc:
                        logger.info(
                            "[FeedbackStore] DatabaseManager unavailable, memory fallback: %s",
                            exc,
                        )
                _feedback_store_singleton = FeedbackStore(db)
    return _feedback_store_singleton
