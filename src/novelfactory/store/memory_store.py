"""MemoryStore — 全局记忆持久化存储（v8.4 落库修复）。

背景: memory 路由历史使用纯内存 dict，进程重启即丢数据。
本模块提供 PostgreSQL 持久化（user_memory / memory_facts 两张表），
无数据库连接时自动回退内存存储，保持既有行为。

用法:
    store = get_memory_store()
    store.save_memory("default", data_dict)
    data = store.load_memory()
    store.create_fact({...})
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class MemoryStore:
    """全局记忆存储 — PostgreSQL 优先，内存回退。"""

    TABLE_MEMORY = "user_memory"
    TABLE_FACTS = "memory_facts"

    def __init__(self, db_manager: Any | None = None) -> None:
        self._db = db_manager
        # 内存回退层（无 DB 或 DB 故障时使用）
        self._mem: dict[str, dict] = {}
        self._facts: dict[str, dict] = {}
        if self._db is not None:
            try:
                self._ensure_schema()
            except Exception as exc:
                logger.warning("[MemoryStore] schema init failed, fallback to memory: %s", exc)
                self._db = None

    @property
    def persistent(self) -> bool:
        """是否已接入 PostgreSQL。"""
        return self._db is not None

    def _ensure_schema(self) -> None:
        assert self._db is not None
        self._db.execute(
            f"CREATE TABLE IF NOT EXISTS {self.TABLE_MEMORY} ("
            "  mem_key TEXT PRIMARY KEY,"
            "  data_json TEXT NOT NULL,"
            "  updated_at TIMESTAMPTZ DEFAULT NOW()"
            ")"
        )
        self._db.execute(
            f"CREATE TABLE IF NOT EXISTS {self.TABLE_FACTS} ("
            "  id TEXT PRIMARY KEY,"
            "  content TEXT NOT NULL,"
            "  category TEXT NOT NULL DEFAULT 'context',"
            "  confidence DOUBLE PRECISION NOT NULL DEFAULT 0.5,"
            "  created_at TIMESTAMPTZ DEFAULT NOW(),"
            "  source TEXT NOT NULL DEFAULT 'unknown'"
            ")"
        )

    # ── memory 主数据 ─────────────────────────────────────────────────────────

    def load_memory(self, key: str = "default") -> dict | None:
        """读取记忆数据，无数据返回 None。"""
        if self._db is not None:
            try:
                row = self._db.fetchone(
                    f"SELECT data_json FROM {self.TABLE_MEMORY} WHERE mem_key = %s",
                    (key,),
                )
                if row:
                    import json

                    return json.loads(row[0])
            except Exception as exc:
                logger.warning("[MemoryStore] load_memory db failed, fallback: %s", exc)
        return self._mem.get(key)

    def save_memory(self, key: str, data: dict) -> None:
        """保存记忆数据（upsert）。"""
        import json

        self._mem[key] = data
        if self._db is None:
            return
        try:
            self._db.execute(
                f"INSERT INTO {self.TABLE_MEMORY} (mem_key, data_json, updated_at) "
                "VALUES (%s, %s, NOW()) "
                "ON CONFLICT (mem_key) DO UPDATE SET data_json = EXCLUDED.data_json, updated_at = NOW()",
                (key, json.dumps(data, ensure_ascii=False)),
            )
        except Exception as exc:
            logger.warning("[MemoryStore] save_memory db failed, kept in memory: %s", exc)

    # ── facts ─────────────────────────────────────────────────────────────────

    def list_facts(self, category: str | None = None) -> list[dict]:
        """列出事实，可按 category 过滤。"""
        if self._db is not None:
            try:
                if category:
                    rows = self._db.fetchall(
                        f"SELECT id, content, category, confidence, created_at, source "
                        f"FROM {self.TABLE_FACTS} WHERE category = %s ORDER BY created_at",
                        (category,),
                    )
                else:
                    rows = self._db.fetchall(
                        f"SELECT id, content, category, confidence, created_at, source "
                        f"FROM {self.TABLE_FACTS} ORDER BY created_at"
                    )
                return [self._row_to_fact(r) for r in rows]
            except Exception as exc:
                logger.warning("[MemoryStore] list_facts db failed, fallback: %s", exc)
        return [dict(f) for f in self._facts.values()]

    def create_fact(self, fact: dict) -> dict:
        """创建事实。"""
        self._facts[fact["id"]] = dict(fact)
        if self._db is None:
            return fact
        try:
            self._db.execute(
                f"INSERT INTO {self.TABLE_FACTS} (id, content, category, confidence, source) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    fact["id"],
                    fact["content"],
                    fact.get("category", "context"),
                    fact.get("confidence", 0.5),
                    fact.get("source", "unknown"),
                ),
            )
        except Exception as exc:
            logger.warning("[MemoryStore] create_fact db failed, kept in memory: %s", exc)
        return fact

    def delete_fact(self, fact_id: str) -> bool:
        """删除事实，返回是否删除。"""
        existed = self._facts.pop(fact_id, None) is not None
        if self._db is not None:
            try:
                result = self._db.execute(
                    f"DELETE FROM {self.TABLE_FACTS} WHERE id = %s", (fact_id,)
                )
                existed = existed or (result is not None and "DELETE" in str(result))
            except Exception as exc:
                logger.warning("[MemoryStore] delete_fact db failed, fallback: %s", exc)
        return existed

    def update_fact(self, fact_id: str, updates: dict) -> dict | None:
        """部分更新事实，返回更新后的事实或 None。"""
        current = self._facts.get(fact_id)
        if current is None and self._db is not None:
            try:
                row = self._db.fetchone(
                    f"SELECT id, content, category, confidence, created_at, source "
                    f"FROM {self.TABLE_FACTS} WHERE id = %s",
                    (fact_id,),
                )
                if row:
                    current = self._row_to_fact(row)
            except Exception as exc:
                logger.warning("[MemoryStore] update_fact db read failed: %s", exc)
        if current is None:
            return None
        merged = {**current, **{k: v for k, v in updates.items() if v is not None}}
        self._facts[fact_id] = merged
        if self._db is not None:
            try:
                self._db.execute(
                    f"UPDATE {self.TABLE_FACTS} SET content = %s, category = %s, confidence = %s "
                    "WHERE id = %s",
                    (merged["content"], merged["category"], merged["confidence"], fact_id),
                )
            except Exception as exc:
                logger.warning("[MemoryStore] update_fact db failed, kept in memory: %s", exc)
        return merged

    def clear(self) -> None:
        """清空所有记忆。"""
        self._mem.clear()
        self._facts.clear()
        if self._db is not None:
            try:
                self._db.execute(f"DELETE FROM {self.TABLE_MEMORY}")
                self._db.execute(f"DELETE FROM {self.TABLE_FACTS}")
            except Exception as exc:
                logger.warning("[MemoryStore] clear db failed: %s", exc)

    @staticmethod
    def _row_to_fact(row: tuple) -> dict:
        return {
            "id": str(row[0]),
            "content": str(row[1]),
            "category": str(row[2]),
            "confidence": float(row[3]),
            "created_at": row[4].isoformat() if hasattr(row[4], "isoformat") else str(row[4]),
            "source": str(row[5]),
        }


# ── 模块级单例 ────────────────────────────────────────────────────────────────

_memory_store_singleton: MemoryStore | None = None
_memory_store_lock = threading.Lock()


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


def get_memory_store() -> MemoryStore:
    """获取共享 MemoryStore 单例（懒加载 DatabaseManager）。"""
    global _memory_store_singleton
    if _memory_store_singleton is None:
        with _memory_store_lock:
            if _memory_store_singleton is None:
                db = None
                if _db_reachable():
                    try:
                        from novelfactory.config.database import DatabaseManager

                        db = DatabaseManager.get_instance()
                    except Exception as exc:
                        logger.info(
                            "[MemoryStore] DatabaseManager unavailable, memory fallback: %s",
                            exc,
                        )
                _memory_store_singleton = MemoryStore(db)
    return _memory_store_singleton
