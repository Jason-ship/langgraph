"""Middleware wrapper — 将中间件链包装到 LangGraph 节点函数上。

LangGraph 不提供全局 before_node/after_node 钩子，
此包装器通过函数装饰器模式实现中间件注入。

支持同步和异步节点函数（v7.0+：异步 Supervisor 节点支持）。
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from novelfactory.middleware.base import MiddlewareChain


def with_middleware(
    node_fn: Callable[..., dict[str, Any]],
    chain: MiddlewareChain,
) -> Callable[..., dict[str, Any]]:
    """将中间件链包装到 LangGraph 节点函数上。

    对于 compiled subgraph（CompiledStateGraph），直接返回原对象，
    LangGraph 内部通过 invoke 调用子图，包装器会破坏调用契约。
    对于普通节点函数，包装 before/after 钩子。
    异步函数（async def）自动使用异步包装器。

    LangGraph 节点函数的签名为 (state, config, **kwargs) → dict。
    """
    # CompiledStateGraph 不能被直接调用，LangGraph 内部使用 invoke
    if isinstance(node_fn, CompiledStateGraph):
        return node_fn

    if inspect.iscoroutinefunction(node_fn):
        return _make_async_wrapper(node_fn, chain)

    return _make_sync_wrapper(node_fn, chain)


def _make_sync_wrapper(
    node_fn: Callable[..., dict[str, Any]],
    chain: MiddlewareChain,
) -> Callable[..., dict[str, Any]]:
    """Create a synchronous middleware wrapper for a sync node function."""

    def wrapped(
        state: dict, config: Optional[RunnableConfig] = None, **kwargs  # noqa: UP045 — intentional, LangGraph type checker requires Optional
    ) -> dict[str, Any]:
        if config is None:
            config = {}

        # Before hooks
        pre_updates = chain.execute_before(state, config)
        if pre_updates:
            state = {**state, **pre_updates}

        # Execute node
        result = node_fn(state, **kwargs)

        # After hooks
        post_updates = chain.execute_after(state, result, config)
        if post_updates:
            result = {**result, **post_updates}

        return result

    return wrapped


def _make_async_wrapper(
    node_fn: Callable[..., dict[str, Any]],
    chain: MiddlewareChain,
) -> Callable[..., dict[str, Any]]:
    """Create an asynchronous middleware wrapper for an async node function."""

    async def async_wrapped(
        state: dict, config: Optional[RunnableConfig] = None, **kwargs  # noqa: UP045 — intentional, LangGraph type checker requires Optional
    ) -> dict[str, Any]:
        if config is None:
            config = {}

        # Before hooks (sync)
        pre_updates = chain.execute_before(state, config)
        if pre_updates:
            state = {**state, **pre_updates}

        # Execute node (awaited)
        result = await node_fn(state, **kwargs)

        # After hooks (sync)
        post_updates = chain.execute_after(state, result, config)
        if post_updates:
            result = {**result, **post_updates}

        return result

    return async_wrapped
