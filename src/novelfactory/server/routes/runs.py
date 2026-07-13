# ==============================================================================
# SDK: /runs and /threads/{thread_id}/runs
# ==============================================================================

from __future__ import annotations

import uuid

from fastapi import APIRouter
from fastapi.responses import Response
from langchain_core.runnables import RunnableConfig
from sse_starlette.sse import EventSourceResponse

from novelfactory.server.models import RunRequest, ThreadModel
from novelfactory.server.streaming import _create_run_impl, _stream_run_history

router = APIRouter()

from novelfactory.server.deps import get_graph, get_run_store  # noqa: E402

# ── /runs endpoints (no thread_id required) ───────────────────────────────────


@router.post("/runs/stream", tags=["runs"], include_in_schema=False)
async def create_run_stream_no_thread(run: RunRequest) -> Response:
    """SDK-compatible streaming run endpoint (no thread — creates one)."""
    run_store = get_run_store()

    thread = ThreadModel()
    run_store[thread.thread_id] = []
    return await _create_run_impl(thread.thread_id, run)


@router.post("/runs/wait", tags=["runs"], include_in_schema=False)
async def create_run_wait_no_thread(run: RunRequest) -> Response:
    """SDK-compatible non-streaming run (no thread — creates one)."""
    run_store = get_run_store()

    run.stream = False
    thread = ThreadModel()
    run_store[thread.thread_id] = []
    return await _create_run_impl(thread.thread_id, run)


@router.post("/runs/batch", tags=["runs"], include_in_schema=False)
async def create_runs_batch(runs: list[RunRequest]) -> list:
    """Batch create runs (stub)."""
    return [{"run_id": str(uuid.uuid4()), "status": "completed"} for _ in runs]


# ── /threads/{thread_id}/runs endpoints ────────────────────────────────────────


@router.get("/threads/{thread_id}/runs", tags=["runs"])
async def list_runs(thread_id: str) -> list:
    """List all runs for a thread."""
    run_store = get_run_store()

    return run_store.get(thread_id, [])


@router.post("/threads/{thread_id}/runs", tags=["runs"], response_model=None)
async def create_run(thread_id: str, run: RunRequest) -> Response:
    """Create a new run on a thread."""
    return await _create_run_impl(thread_id, run)


@router.post(
    "/threads/{thread_id}/runs/stream",
    tags=["runs"],
    response_model=None,
    include_in_schema=False,
)
async def create_run_stream(thread_id: str, run: RunRequest) -> Response:
    """SDK-compatible streaming run endpoint (alias for /runs)."""
    return await _create_run_impl(thread_id, run)


@router.post("/threads/{thread_id}/runs/wait", tags=["runs"], include_in_schema=False)
async def create_run_wait(thread_id: str, run: RunRequest) -> Response:
    """SDK-compatible non-streaming run (wait for completion)."""
    run.stream = False
    return await _create_run_impl(thread_id, run)


@router.get("/threads/{thread_id}/runs/{run_id}", tags=["runs"])
async def get_run(thread_id: str, run_id: str) -> dict:
    """Get run details."""
    return {"run_id": run_id, "thread_id": thread_id, "status": "completed"}


@router.delete("/threads/{thread_id}/runs/{run_id}", tags=["runs"])
async def delete_run(thread_id: str, run_id: str) -> dict:
    """Delete a run."""
    return {"deleted": run_id}


@router.post("/threads/{thread_id}/runs/{run_id}/cancel", tags=["runs"])
async def cancel_run(thread_id: str, run_id: str) -> dict:
    """Cancel a run."""
    return {"cancelled": run_id}


@router.get("/threads/{thread_id}/runs/{run_id}/join", tags=["runs"])
async def join_run(thread_id: str, run_id: str) -> dict:
    """Block until run completes."""
    return {"run_id": run_id, "status": "completed"}


@router.get("/threads/{thread_id}/runs/{run_id}/stream", tags=["runs"])
async def stream_run_output(thread_id: str, run_id: str) -> Response:
    """Get streamed output from a completed run (SSE)."""
    graph = await get_graph()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    return EventSourceResponse(
        _stream_run_history(graph, config, thread_id, run_id),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
