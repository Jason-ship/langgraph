"""Shared runtime dependencies for route handlers.

Breaks the circular import: app.py → routes → app.py.

Route modules import these accessors at module level without triggering
a circular import — the actual ``get_app`` / ``_run_store`` resolution
happens lazily at call time inside each accessor.

Usage in route files::

    from novelfactory.server.deps import get_graph, get_run_store, get_store

    @router.get("/example")
    async def example():
        graph = await get_graph()
        store = await get_store()
        run_store = get_run_store()
"""

from __future__ import annotations

from typing import Any

from langgraph.graph.state import CompiledStateGraph as CompiledGraph
from langgraph.store.postgres.aio import AsyncPostgresStore

from novelfactory.server.graph_router import GraphRouter


async def get_graph() -> Any:
    """Get the compiled graph singleton.

    Lazy-imports ``get_app`` from ``server.app`` at call time to avoid
    the module-load circular dependency.
    """
    from novelfactory.server.app import get_app

    return await get_app()


async def get_store() -> AsyncPostgresStore:
    """Get the LangGraph ``AsyncPostgresStore`` attached to the graph singleton.

    Raises:
        RuntimeError: if the graph has no store attached.
    """
    from novelfactory.server.app import get_app

    graph = await get_app()
    store = getattr(graph, "store", None)
    if store is None:
        raise RuntimeError(
            "Store not available — graph has no store attached. "
            "Ensure the graph was compiled with a store."
        )
    return store


def get_run_store() -> dict:
    """Get the ephemeral run store (``thread_id → list[run_record]``).

    This is the in-memory dict that tracks active/recent runs for SSE
    streaming and status queries.
    """
    from novelfactory.server.app import _run_store

    return _run_store


async def get_lead_graph() -> CompiledGraph:
    """Get the lead agent conversational graph.

    Lazy-imports from server.app at call time.
    """
    from novelfactory.server.app import get_lead_graph as _get_lead_graph

    return await _get_lead_graph()


async def get_graph_router() -> GraphRouter:
    """Get the GraphRouter that dispatches between batch and lead graphs."""
    from novelfactory.server.app import get_router

    return await get_router()


# ── 通用依赖注入工厂（参考 DeerFlow deps.py _require 模式） ──────────────

from fastapi import HTTPException, Request


def _require(attr: str, label: str):
    """工厂函数，生成 FastAPI 依赖，从 app.state 获取单例。

    使用方式:
        from fastapi import Depends
        get_channel_service = _require("channel_service", "Channel service")

        @router.get("/channels")
        async def list_channels(service=Depends(get_channel_service)):
            ...
    """
    async def dependency(request: Request):
        value = getattr(request.app.state, attr, None)
        if value is None:
            raise HTTPException(status_code=503, detail=f"{label} not available")
        return value
    return dependency
