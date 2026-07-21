"""Regenerate API — prepare a thread for message regeneration.

DeerFlow frontend calls this endpoint before regenerating a message
to obtain the checkpoint info needed to restore the target state.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from novelfactory.server.deps import get_graph

logger = logging.getLogger(__name__)
router = APIRouter(tags=["threads"])


class RegeneratePrepareRequest(BaseModel):
    """Request to prepare a thread for regeneration."""

    message_id: str | None = None
    checkpoint_id: str | None = None


@router.post("/threads/{thread_id}/runs/regenerate/prepare")
async def prepare_regenerate(
    thread_id: str, req: RegeneratePrepareRequest
) -> dict:
    """Prepare a thread for message regeneration.

    This endpoint is called before regenerating a message.
    It returns the checkpoint info needed to restore the state.
    """
    graph = await get_graph()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    try:
        state = await graph.aget_state(config)
        checkpoint = getattr(state, "checkpoint", {}) or {}

        return {
            "status": "ready",
            "thread_id": thread_id,
            "checkpoint_id": checkpoint.get("checkpoint_id"),
            "parent_checkpoint_id": checkpoint.get("parent_checkpoint_id"),
            "next": list(state.next) if state.next else [],
        }
    except Exception as e:
        raise HTTPException(
            status_code=404, detail=f"Thread {thread_id} not found"
        ) from e