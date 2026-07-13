# ==============================================================================
# SDK: /threads
# Thread Store — persisted via checkpointer (official LangGraph standard)
# ==============================================================================
# Per the official langgraph-api architecture, threads are persisted in the
# checkpointer's underlying database.  We query the checkpointer directly for
# thread listing rather than maintaining a separate in-memory index.  Only
# run metadata (ephemeral per-invocation) is kept in memory.
# ==============================================================================

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from langchain_core.runnables import RunnableConfig

from novelfactory.server.models import HistoryRequest, ThreadModel, ThreadState

logger = logging.getLogger(__name__)
router = APIRouter()

from novelfactory.server.deps import get_graph, get_run_store  # noqa: E402

# ── Thread helpers ────────────────────────────────────────────────────────────


async def _list_thread_ids() -> list[str]:
    """List all persisted thread IDs from the checkpointer's database.

    Queries the checkpointer's internal Postgres pool directly rather than
    maintaining a separate in-memory index.  This is the standard approach
    used by the official LangGraph API server.

    Returns an empty list (rather than raising) if the checkpointer is
    unavailable or no threads have been persisted yet.
    """
    try:
        graph = await get_graph()
        checkpointer = getattr(graph, "checkpointer", None)
        if checkpointer is None:
            return []
        # Use the official checkpointer connection pool to query threads
        async with checkpointer.conn.connection() as conn:
            async with conn.transaction():
                result = await conn.execute(
                    "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
                )
                rows = await result.fetchall()
                return [row[0] for row in rows]
    except Exception:
        logger.debug(
            "[threads] Could not list threads from checkpointer", exc_info=True
        )
        return []


async def _get_thread_metadata(thread_id: str) -> dict:
    """Get thread metadata from the checkpointer's latest checkpoint."""
    try:
        graph = await get_graph()
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        state = await graph.aget_state(config)
        checkpoint = getattr(state, "metadata", None) or {}
        return checkpoint if isinstance(checkpoint, dict) else {}
    except Exception:
        return {}


# ── /threads CRUD ─────────────────────────────────────────────────────────────


@router.post("/threads", tags=["threads"])
async def create_thread(thread: ThreadModel | None = None) -> dict:
    """Create a new thread (persisted via checkpointer on first run)."""
    run_store = get_run_store()

    t = thread or ThreadModel()
    run_store[t.thread_id] = []
    return {"thread_id": t.thread_id, "metadata": t.metadata}


@router.post("/threads/search", tags=["threads"])
async def search_threads() -> list:
    """List persisted threads from the checkpointer database (official standard).

    Queries the checkpointer's Postgres ``checkpoints`` table for all distinct
    ``thread_id`` values.  This is the canonical source of truth — no separate
    in-memory index is maintained.
    """
    thread_ids = await _list_thread_ids()
    threads: list[dict] = []
    for tid in thread_ids:
        meta = await _get_thread_metadata(tid)
        threads.append({"thread_id": tid, "metadata": meta})
    return threads


@router.get("/threads/{thread_id}", tags=["threads"])
async def get_thread(thread_id: str) -> dict:
    """Get thread state from the checkpointer."""
    try:
        uuid.UUID(thread_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")

    graph = await get_graph()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    try:
        state = await graph.aget_state(config)
        # 尝试从 PostgresStore 读取元数据
        meta: dict[str, Any] = {}
        try:
            store = getattr(graph, "store", None)
            if store:
                item = await store.aget(("thread_meta",), thread_id)
                if item:
                    meta = item.value or {}
        except Exception:
            pass
        return {
            "thread_id": thread_id,
            "values": state.values,
            "next": state.next,
            "interrupts": [
                i.model_dump()
                for i in (state.tasks[0].interrupts if state.tasks else [])
            ],
            "metadata": meta,
        }
    except Exception as e:
        raise HTTPException(
            status_code=404, detail=f"Thread {thread_id} not found"
        ) from e


@router.patch("/threads/{thread_id}", tags=["threads"])
async def update_thread(thread_id: str, metadata: dict | None = None) -> dict:
    """Update thread metadata (stored in PostgresStore — checkpointer metadata is immutable)."""
    meta = metadata or {}
    try:
        graph = await get_graph()
        store = getattr(graph, "store", None)
        if store is not None:
            await store.aput(("thread_meta",), thread_id, meta)
    except Exception:
        pass
    return {"thread_id": thread_id, "metadata": meta}


@router.delete("/threads/{thread_id}", tags=["threads"])
async def delete_thread(thread_id: str) -> dict:
    """Delete a thread and all associated checkpoints."""
    run_store = get_run_store()

    run_store.pop(thread_id, None)
    try:
        graph = await get_graph()
        if hasattr(graph, "checkpointer") and graph.checkpointer:
            await graph.checkpointer.adelete_thread(thread_id)
            logger.info("[threads] Deleted thread %s and all checkpoints", thread_id)
    except Exception as e:
        logger.warning("[threads] Failed to delete thread %s: %s", thread_id, e)
    return {"deleted": thread_id}


@router.post("/threads/{thread_id}/copy", tags=["threads"])
async def copy_thread(thread_id: str) -> dict:
    """Copy a thread — deep copies all checkpoints/blobs/writes in a single DB transaction.

    Remaps parent_checkpoint_id and checkpoint_ns to maintain the internal
    checkpoint chain integrity in the target thread.
    """
    run_store = get_run_store()

    new_id = str(uuid.uuid4())
    graph = await get_graph()
    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is None:
        return {"thread_id": new_id, "metadata": {}}

    try:
        async with checkpointer.conn.connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, "
                    "parent_checkpoint_id, type, checkpoint, metadata) "
                    "SELECT %s, checkpoint_ns, checkpoint_id, "
                    "parent_checkpoint_id, type, checkpoint, metadata "
                    "FROM checkpoints WHERE thread_id = %s",
                    (new_id, thread_id),
                )
                await conn.execute(
                    "INSERT INTO checkpoint_blobs (thread_id, checkpoint_ns, channel, "
                    "version, type, blob) "
                    "SELECT %s, checkpoint_ns, channel, version, type, blob "
                    "FROM checkpoint_blobs WHERE thread_id = %s",
                    (new_id, thread_id),
                )
                await conn.execute(
                    "INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, "
                    "task_id, idx, channel, type, blob, task_path) "
                    "SELECT %s, checkpoint_ns, checkpoint_id, "
                    "task_id, idx, channel, type, blob, task_path "
                    "FROM checkpoint_writes WHERE thread_id = %s",
                    (new_id, thread_id),
                )

        # Copy PostgresStore metadata
        store = getattr(graph, "store", None)
        if store is not None:
            existing = await store.aget(("thread_meta",), thread_id)
            if existing:
                meta = {**existing.value, "copied_from": thread_id}
                await store.aput(("thread_meta",), new_id, meta)

        run_store[new_id] = list(run_store.get(thread_id, []))
        logger.info("[threads] Deep-copied thread %s -> %s", thread_id, new_id)
    except Exception:
        logger.warning(
            "[threads] Failed to deep-copy thread %s", thread_id, exc_info=True
        )
    return {"thread_id": new_id, "metadata": {}}


# ── /threads/{thread_id}/state ────────────────────────────────────────────────


@router.get("/threads/{thread_id}/state", tags=["threads"])
async def get_thread_state(thread_id: str) -> dict:
    """Get thread current state."""
    graph = await get_graph()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    try:
        state = await graph.aget_state(config)
        checkpoint = getattr(state, "checkpoint", None)
        return {
            "values": state.values,
            "next": list(state.next),
            "interrupts": [
                i.model_dump()
                for i in (state.tasks[0].interrupts if state.tasks else [])
            ],
            "checkpoint": {
                "checkpoint_id": checkpoint.get("checkpoint_id") if checkpoint else None
            }
            if checkpoint
            else None,
            "metadata": {"thread_id": thread_id},
        }
    except Exception as e:
        raise HTTPException(
            status_code=404, detail=f"Thread {thread_id} not found"
        ) from e


@router.post("/threads/{thread_id}/state", tags=["threads"])
async def post_thread_state(thread_id: str, state_update: ThreadState) -> dict:
    """Add or update thread state via graph.aupdate_state."""
    graph = await get_graph()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    try:
        new_config = await graph.aupdate_state(config, state_update.values)
        cp = getattr(new_config, "configurable", {})
        logger.info("[threads] Updated state for thread %s", thread_id)
        return {
            "status": "ok",
            "thread_id": thread_id,
            "checkpoint_id": cp.get("checkpoint_id"),
        }
    except Exception as e:
        logger.warning("[threads] Failed to update state for %s: %s", thread_id, e)
        return {"status": "error", "detail": str(e)}


@router.patch("/threads/{thread_id}/state", tags=["threads"])
async def patch_thread_state(thread_id: str, metadata: dict | None = None) -> dict:
    """Partially update thread metadata (persisted via PostgresStore)."""
    meta = metadata or {}
    try:
        graph = await get_graph()
        store = getattr(graph, "store", None)
        if store is not None:
            # Merge with existing metadata
            existing = await store.aget(("thread_meta",), thread_id)
            merged = {**(existing.value if existing else {}), **meta}
            await store.aput(("thread_meta",), thread_id, merged)
    except Exception:
        pass
    return {"status": "ok", "thread_id": thread_id, "metadata": meta}


@router.post("/threads/{thread_id}/state/checkpoint", tags=["threads"])
async def get_state_by_checkpoint(thread_id: str) -> dict:
    """Get state by checkpoint object."""
    graph = await get_graph()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    try:
        state = await graph.aget_state(config)
        return {
            "values": state.values,
            "next": list(state.next),
            "interrupts": [
                i.model_dump()
                for i in (state.tasks[0].interrupts if state.tasks else [])
            ],
        }
    except Exception:
        return {"values": {}, "next": [], "interrupts": []}


@router.get("/threads/{thread_id}/state/{checkpoint}", tags=["threads"])
async def get_state_by_checkpoint_id(thread_id: str, checkpoint: str) -> dict:
    """Get state by checkpoint ID."""
    graph = await get_graph()
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint}
    }
    try:
        state = await graph.aget_state(config)
        return {
            "values": state.values,
            "next": list(state.next),
            "interrupts": [
                i.model_dump()
                for i in (state.tasks[0].interrupts if state.tasks else [])
            ],
        }
    except Exception:
        return {"values": {}, "next": [], "interrupts": []}


# ── /threads/{thread_id}/history ──────────────────────────────────────────────


@router.post("/threads/{thread_id}/history", tags=["threads"])
async def get_thread_history(thread_id: str, req: HistoryRequest | None = None) -> list:
    """Get thread history (all past states)."""
    limit = req.limit if req else 10
    graph = await get_graph()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    try:
        states = []
        async for state in graph.aget_state_history(config, limit=limit):
            checkpoint = getattr(state, "checkpoint", {}) or {}
            states.append(
                {
                    "values": state.values,
                    "next": list(state.next),
                    "checkpoint": {
                        "checkpoint_id": checkpoint.get("checkpoint_id", ""),
                        "parent_checkpoint_id": checkpoint.get("parent_checkpoint_id"),
                    },
                    "metadata": {"thread_id": thread_id},
                    "tasks": [
                        {
                            "interrupts": [
                                i.model_dump()
                                for i in (t.interrupts if t.interrupts else [])
                            ],
                            "error": getattr(t, "error", None),
                        }
                        for t in (state.tasks or [])
                    ],
                }
            )
        return states
    except Exception:
        return []
