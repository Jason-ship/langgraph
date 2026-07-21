"""Thread branching API — create branches from specific checkpoints.

DeerFlow frontend uses this to create alternative conversation paths
by copying all checkpoints up to a user-specified point.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from novelfactory.server.deps import get_graph

logger = logging.getLogger(__name__)
router = APIRouter(tags=["threads"])


class BranchRequest(BaseModel):
    """Request to branch a thread from a specific message turn."""

    message_id: str
    message_ids: list[str] = []
    title: str | None = None


@router.post("/threads/{thread_id}/branches")
async def branch_thread(thread_id: str, req: BranchRequest) -> dict:
    """Create a branch from a specific checkpoint in the thread.

    Creates a new thread with all checkpoints up to the specified point.
    This allows users to explore alternative conversation paths.
    """
    graph = await get_graph()
    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is None:
        raise HTTPException(status_code=400, detail="Checkpointer not available")

    new_thread_id = str(uuid.uuid4())

    # Map frontend fields → internal fields
    from_message_id = req.message_id or (req.message_ids[0] if req.message_ids else None)

    # Determine the checkpoint_id to branch from
    checkpoint_id = None
    if from_message_id:
        checkpoint_id = await _resolve_checkpoint_from_message(
            checkpointer, thread_id, from_message_id
        )

    if not checkpoint_id:
        # If no checkpoint specified, copy all checkpoints (full branch)
        checkpoint_id = None

    try:
        async with checkpointer.conn.connection() as conn:
            async with conn.transaction():
                if checkpoint_id:
                    # Copy checkpoints up to the specified point using recursive CTE
                    # to walk the parent_checkpoint_id chain (UUID <= comparison is
                    # unreliable for checkpoint ordering).
                    await conn.execute(
                        "WITH RECURSIVE checkpoint_chain AS ("
                        "  SELECT checkpoint_id, parent_checkpoint_id "
                        "  FROM checkpoints "
                        "  WHERE thread_id = %s AND checkpoint_id = %s::uuid "
                        "  UNION ALL "
                        "  SELECT c.checkpoint_id, c.parent_checkpoint_id "
                        "  FROM checkpoints c "
                        "  INNER JOIN checkpoint_chain cc "
                        "    ON c.checkpoint_id = cc.parent_checkpoint_id "
                        "  WHERE c.thread_id = %s"
                        ") "
                        "INSERT INTO checkpoints "
                        "(thread_id, checkpoint_ns, checkpoint_id, "
                        "parent_checkpoint_id, type, checkpoint, metadata) "
                        "SELECT %s, cp.checkpoint_ns, cp.checkpoint_id, "
                        "cp.parent_checkpoint_id, cp.type, cp.checkpoint, cp.metadata "
                        "FROM checkpoints cp "
                        "WHERE cp.thread_id = %s "
                        "AND cp.checkpoint_id IN "
                        "  (SELECT checkpoint_id FROM checkpoint_chain)",
                        (thread_id, checkpoint_id, thread_id,
                         new_thread_id, thread_id),
                    )
                else:
                    # Copy all checkpoints
                    await conn.execute(
                        "INSERT INTO checkpoints "
                        "(thread_id, checkpoint_ns, checkpoint_id, "
                        "parent_checkpoint_id, type, checkpoint, metadata) "
                        "SELECT %s, checkpoint_ns, checkpoint_id, "
                        "parent_checkpoint_id, type, checkpoint, metadata "
                        "FROM checkpoints WHERE thread_id = %s",
                        (new_thread_id, thread_id),
                    )

                # Copy checkpoint_blobs
                if checkpoint_id:
                    await conn.execute(
                        "INSERT INTO checkpoint_blobs "
                        "(thread_id, checkpoint_ns, channel, version, type, blob) "
                        "SELECT %s, checkpoint_ns, channel, version, type, blob "
                        "FROM checkpoint_blobs WHERE thread_id = %s "
                        "AND checkpoint_id IN "
                        "  (SELECT checkpoint_id FROM checkpoint_chain)",
                        (new_thread_id, thread_id),
                    )
                else:
                    await conn.execute(
                        "INSERT INTO checkpoint_blobs "
                        "(thread_id, checkpoint_ns, channel, version, type, blob) "
                        "SELECT %s, checkpoint_ns, channel, version, type, blob "
                        "FROM checkpoint_blobs WHERE thread_id = %s",
                        (new_thread_id, thread_id),
                    )

                # Copy checkpoint_writes
                if checkpoint_id:
                    await conn.execute(
                        "INSERT INTO checkpoint_writes "
                        "(thread_id, checkpoint_ns, checkpoint_id, "
                        "task_id, idx, channel, type, blob, task_path) "
                        "SELECT %s, checkpoint_ns, checkpoint_id, "
                        "task_id, idx, channel, type, blob, task_path "
                        "FROM checkpoint_writes WHERE thread_id = %s "
                        "AND checkpoint_id IN "
                        "  (SELECT checkpoint_id FROM checkpoint_chain)",
                        (new_thread_id, thread_id),
                    )
                else:
                    await conn.execute(
                        "INSERT INTO checkpoint_writes "
                        "(thread_id, checkpoint_ns, checkpoint_id, "
                        "task_id, idx, channel, type, blob, task_path) "
                        "SELECT %s, checkpoint_ns, checkpoint_id, "
                        "task_id, idx, channel, type, blob, task_path "
                        "FROM checkpoint_writes WHERE thread_id = %s",
                        (new_thread_id, thread_id),
                    )

        # Copy store metadata
        try:
            store = getattr(graph, "store", None)
            if store is not None:
                existing = await store.aget(("thread_meta",), thread_id)
                if existing and existing.value:
                    meta = dict(existing.value)
                    meta["branched_from"] = thread_id
                    await store.aput(("thread_meta",), new_thread_id, meta)
        except Exception:
            logger.debug("[branches] Store metadata copy skipped")

        logger.info(
            "[branches] Thread %s branched -> %s (checkpoint: %s)",
            thread_id,
            new_thread_id,
            checkpoint_id or "all",
        )

        return {
            "thread_id": new_thread_id,
            "parent_thread_id": thread_id,
            "parent_checkpoint_id": checkpoint_id or "",
            "branched_from_message_id": from_message_id or "",
            "workspace_clone_mode": "none",
        }

    except Exception as e:
        logger.exception("[branches] Failed to branch thread %s", thread_id)
        raise HTTPException(
            status_code=500, detail=f"Failed to branch thread: {e}"
        ) from e


async def _resolve_checkpoint_from_message(
    checkpointer: Any, thread_id: str, message_id: str
) -> str | None:
    """Resolve a checkpoint_id from a message_id by searching checkpoint metadata."""
    try:
        async with checkpointer.conn.connection() as conn:
            async with conn.transaction():
                # Look for checkpoint metadata containing the message_id
                result = await conn.execute(
                    "SELECT checkpoint_id, metadata "
                    "FROM checkpoints "
                    "WHERE thread_id = %s "
                    "AND metadata::text LIKE %s "
                    "ORDER BY checkpoint_id DESC "
                    "LIMIT 1",
                    (thread_id, f"%{message_id}%"),
                )
                row = await result.fetchone()
                if row:
                    return str(row[0])
    except Exception as e:
        logger.debug(
            "[branches] Failed to resolve checkpoint from message: %s", e
        )
    return None