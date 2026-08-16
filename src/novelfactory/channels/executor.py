"""RunExecutor — 通道层执行器抽象（v8.4）。

ChannelManager 历史直接调用 LangGraph graph.ainvoke/astream_events，
与本项目运行时强耦合。本模块把"执行模型"抽象为接口：

- RunExecutor: 通道层唯一依赖的执行协议
- LangGraphRunExecutor: 默认实现（包装 CompiledStateGraph）

后续 DSH 桥接可实现 HTTP RunExecutor（经 REST 调 NovelFactory 服务），
ChannelManager 无需改动即可切换执行后端。
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Protocol

logger = logging.getLogger(__name__)


class AgentUnavailableError(RuntimeError):
    """执行后端不可用（graph 未初始化/未注入）。

    v8.4-r (Review 修复): ChannelManager 通过捕获本异常向用户
    发送明确错误消息，替代脆弱的异常消息字符串匹配。
    """


class RunExecutor(Protocol):
    """Channel layer's execution contract (graph-agnostic)."""

    async def ainvoke(
        self,
        messages: dict[str, Any],
        *,
        thread_id: str,
        recursion_limit: int,
        context: dict[str, Any],
    ) -> Any:
        """Run one turn synchronously (non-streaming), return final result."""
        ...

    async def astream_events(
        self,
        messages: dict[str, Any],
        *,
        thread_id: str,
        recursion_limit: int,
        context: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream turn events (v2 event stream), yielding LangGraph events."""
        ...


class LangGraphRunExecutor:
    """Default RunExecutor backed by a compiled LangGraph state graph.

    Args:
        get_graph: zero-arg callable returning the compiled graph
            (kept lazy to preserve existing DI pattern).
        get_context: optional callable(thread_id) -> context dict
            (NovelContext for conversational graph).
    """

    def __init__(
        self,
        get_graph: Any = None,
        get_context: Any = None,
        default_recursion_limit: int = 25,
    ) -> None:
        self._get_graph = get_graph
        self._get_context = get_context
        self._default_recursion_limit = default_recursion_limit

    def _resolve_graph(self) -> Any:
        graph = self._get_graph() if self._get_graph else None
        if graph is None:
            raise AgentUnavailableError("Agent graph not available")
        return graph

    def _resolve_context(self, thread_id: str, fallback: dict[str, Any]) -> dict[str, Any]:
        if self._get_context:
            ctx = self._get_context(thread_id)
            if ctx is not None:
                return ctx
        return fallback

    async def ainvoke(
        self,
        messages: dict[str, Any],
        *,
        thread_id: str,
        recursion_limit: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> Any:
        graph = self._resolve_graph()
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": recursion_limit
            if recursion_limit is not None
            else self._default_recursion_limit,
        }
        ctx = self._resolve_context(thread_id, context or {"thread_id": thread_id})
        return await graph.ainvoke(messages, config=config, context=ctx)

    async def astream_events(
        self,
        messages: dict[str, Any],
        *,
        thread_id: str,
        recursion_limit: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        graph = self._resolve_graph()
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": recursion_limit
            if recursion_limit is not None
            else self._default_recursion_limit,
        }
        ctx = self._resolve_context(thread_id, context or {"thread_id": thread_id})
        async for event in graph.astream_events(
            messages,
            config=config,
            version="v2",
            context=ctx,
        ):
            yield event
