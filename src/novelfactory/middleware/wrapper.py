"""Middleware wrapper — 将中间件链包装到 LangGraph 节点函数上。

LangGraph 不提供全局 before_node/after_node 钩子，
此包装器通过函数装饰器模式实现中间件注入。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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

    LangGraph 节点函数的签名为 (state, config, **kwargs) → dict。
    """
    # CompiledStateGraph 不能被直接调用，LangGraph 内部使用 invoke
    if isinstance(node_fn, CompiledStateGraph):
        return node_fn

    def wrapped(state: dict, config: dict | None = None, **kwargs) -> dict[str, Any]:
        if config is None:
            config = {}

        # Before hooks
        pre_updates = chain.execute_before(state, config)
        if pre_updates:
            state = {**state, **pre_updates}

        # Execute node
        result = node_fn(state, config, **kwargs)

        # After hooks
        post_updates = chain.execute_after(state, result, config)
        if post_updates:
            result = {**result, **post_updates}

        return result

    return wrapped
