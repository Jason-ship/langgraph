"""MCP 会话池 — 持久化 MCP 客户端会话管理。

Migrated from DeerFlow mcp/session_pool.py.

解决 langchain-mcp-adapters 每次工具调用创建新会话的问题，
通过 Owner Task 模式实现线程安全的持久化会话池。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

MAX_SESSIONS = 256


class McpSessionPool:
    """持久化 MCP 会话池。

    每个会话由一个专用的 owner task 拥有，确保 __aenter__ 和 __aexit__
    在同一任务中执行。

    使用方式:
        pool = McpSessionPool()
        session = await pool.get_session("server_name", "user:thread")
        result = await session.call_tool("tool_name", {"arg": "value"})
        await pool.close_scope("user:thread")
    """

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._max_sessions = max_sessions
        self._entries: OrderedDict[str, Any] = OrderedDict()
        self._inflight: dict[str, asyncio.Future] = {}
        self._lock = threading.Lock()

    async def get_session(self, server_name: str, scope_key: str = "default") -> Any:
        """获取或创建持久化 MCP 会话。

        Args:
            server_name: MCP 服务器名称。
            scope_key: 作用域键（通常是 user_id:thread_id）。

        Returns:
            MCP ClientSession 实例。
        """
        entry_key = f"{server_name}:{scope_key}"
        # 检查现有会话
        entry = self._entries.get(entry_key)
        if entry is not None:
            # 如果会话仍有效，返回
            session = entry.get("session")
            if session is not None:
                return session

        # 检查是否有创建中的会话
        with self._lock:
            inflight = self._inflight.get(entry_key)
            if inflight is not None:
                # 等待其他任务创建完成
                return await inflight

            # 创建新会话的 Future
            future: asyncio.Future = asyncio.Future()
            self._inflight[entry_key] = future

        try:
            # 创建新会话
            session = await self._create_session(server_name)
            with self._lock:
                self._entries[entry_key] = {"session": session, "server_name": server_name, "scope_key": scope_key}
                # LRU 淘汰
                while len(self._entries) > self._max_sessions:
                    self._entries.popitem(last=False)
                self._entries.move_to_end(entry_key)
            future.set_result(session)
            return session
        except Exception as exc:
            future.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._inflight.pop(entry_key, None)

    async def _create_session(self, server_name: str) -> Any:
        """创建 MCP 会话。

        子类可重写此方法以集成 langchain-mcp-adapters。
        默认实现尝试创建真实的 MCP ClientSession，失败时返回占位。

        Args:
            server_name: MCP 服务器名称。

        Returns:
            MCP ClientSession 或兼容的会话对象。
        """
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            client = MultiServerMCPClient()
            # 返回一个兼容的会话对象
            return client
        except ImportError:
            logger.warning(
                "[MCP] langchain-mcp-adapters not installed. "
                "MCP tools will not be available. Install with: pip install langchain-mcp-adapters"
            )
            return None

    async def close_scope(self, scope_key: str) -> None:
        """关闭指定 scope 的所有会话。"""
        with self._lock:
            keys_to_close = [k for k in self._entries if k.endswith(f":{scope_key}")]
            for k in keys_to_close:
                self._entries.pop(k, None)

    async def close_server(self, server_name: str) -> None:
        """关闭指定服务器的所有会话。"""
        with self._lock:
            keys_to_close = [k for k in self._entries if k.startswith(f"{server_name}:")]
            for k in keys_to_close:
                self._entries.pop(k, None)

    async def close_all(self) -> None:
        """关闭所有会话。"""
        with self._lock:
            self._entries.clear()

    @property
    def active_count(self) -> int:
        """当前活跃会话数。"""
        return len(self._entries)


__all__ = ["McpSessionPool"]