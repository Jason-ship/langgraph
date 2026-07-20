"""Agent Memory 系统 — 插件化记忆管理。

Migrated from DeerFlow agents/memory/manager.py + agents/memory/tools.py.

提供可插拔的记忆后端架构，支持不同的记忆存储实现。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class MemoryManager(ABC):
    """记忆管理器抽象基类。

    子类实现具体的记忆后端（内存、向量数据库、图数据库等）。
    """

    @abstractmethod
    async def add(self, thread_id: str, messages: list[Any], **kwargs: Any) -> None:
        """写入记忆。"""
        ...

    @abstractmethod
    async def get_context(self, thread_id: str, **kwargs: Any) -> str:
        """获取记忆上下文（用于注入 prompt）。"""
        ...

    @abstractmethod
    async def search(self, query: str, limit: int = 5, **kwargs: Any) -> list[dict[str, Any]]:
        """搜索记忆。"""
        ...


class InMemoryMemoryManager(MemoryManager):
    """内存记忆管理器（简化版实现）。"""

    def __init__(self) -> None:
        self._memories: dict[str, list[dict[str, Any]]] = {}

    async def add(self, thread_id: str, messages: list[Any], **kwargs: Any) -> None:
        if thread_id not in self._memories:
            self._memories[thread_id] = []
        for msg in messages:
            content = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
            role = getattr(msg, "type", "") if not isinstance(msg, dict) else msg.get("type", "")
            self._memories[thread_id].append({"role": role, "content": str(content)[:500]})
        # 保留最近 50 条
        self._memories[thread_id] = self._memories[thread_id][-50:]

    async def get_context(self, thread_id: str, **kwargs: Any) -> str:
        memories = self._memories.get(thread_id, [])
        if not memories:
            return ""
        lines = ["<memory>", "Previous conversation summary:", ""]
        for m in memories[-10:]:  # 最近 10 条
            role = m.get("role", "unknown")
            content = m.get("content", "")
            if content:
                lines.append(f"[{role}]: {content[:200]}")
        lines.append("</memory>")
        return "\n".join(lines)

    async def search(self, query: str, limit: int = 5, **kwargs: Any) -> list[dict[str, Any]]:
        # 简化版：返回最近 N 条
        all_memories = []
        for memories in self._memories.values():
            all_memories.extend(memories)
        return all_memories[-limit:]


# 全局单例
_memory_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    """获取全局记忆管理器单例。"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = InMemoryMemoryManager()
    return _memory_manager


def set_memory_manager(manager: MemoryManager) -> None:
    """设置全局记忆管理器。"""
    global _memory_manager
    _memory_manager = manager


__all__ = [
    "MemoryManager",
    "InMemoryMemoryManager",
    "get_memory_manager",
    "set_memory_manager",
]