# ==============================================================================
# SSE Helpers & Streaming Functions
# ==============================================================================

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

if TYPE_CHECKING:
    from exceptiongroup import BaseExceptionGroup

try:
    from exceptiongroup import BaseExceptionGroup  # Python 3.10 compat
except ImportError:
    BaseExceptionGroup = type("BaseExceptionGroup", (Exception,), {})
from fastapi.responses import Response
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph as CompiledGraph
from langgraph.types import Command
from sse_starlette.sse import EventSourceResponse

from novelfactory.server.models import RunRequest
from novelfactory.server.stream_tracker import StreamStateTracker
from novelfactory.state.novel_context import NovelContext
from novelfactory.utils.message_utils import classify_message_type

logger = logging.getLogger(__name__)


# ── SSE Helpers ──────────────────────────────────────────────────────────────


def _format_sse(event: str, data: Any) -> dict:
    """Format a Server-Sent Event message (sse-starlette compatible)."""
    return {"event": event, "data": json.dumps(data, default=str)}


async def _detect_stream_interrupts(
    graph: CompiledGraph,
    config: RunnableConfig,
    thread_id: str,
) -> list[dict]:
    """Check for interrupts after stream completes."""
    final_state = await graph.aget_state(config)
    if not final_state.next:
        return []
    interrupts = []
    for task in final_state.tasks:
        for i in task.interrupts:
            interrupts.append({"value": i.value, "id": getattr(i, "id", None)})
    if not interrupts:
        return []
    return [
        _format_sse(
            "interrupt",
            {
                "thread_id": thread_id,
                "interrupts": interrupts,
                "next": list(final_state.next),
            },
        ),
    ]


def _process_stream_event(event: dict) -> list[dict]:
    """Process a single ``astream_events(version=\"v2\")`` event into SDK-compatible SSE.

    Maps LangGraph's internal ``StreamPart`` types to the SSE wire format expected by
    ``@langchain/langgraph-sdk`` (specifically ``useStream()`` in ``stream.js``).

    Official LangGraph ``StreamPart`` types (types.py L262-L352):
      - ``ValuesStreamPart``: ``{type: "values", ns: (...), data: StateT, interrupts: (...)}``
      - ``MessagesStreamPart``: ``{type: "messages", ns: (...), data: (AnyMessage, dict)}``
      - ``UpdatesStreamPart``: ``{type: "updates", ns: (...), data: dict[str, Any]}``

    SDK wire format (useStream handler, stream.js L370-411):
      - ``values`` → ``setStreamValues(data)`` where ``data`` is the full state dict
      - ``messages`` → ``const [serialized, _meta] = data`` (2-tuple: message + metadata)
      - ``updates`` → ``onUpdateEvent(data)`` where ``data`` is ``{nodeName: updateDict}``
      - ``end`` → signals stream termination
    """
    event_type = event.get("type", "")
    data = event.get("data", {})

    if event_type == "values":
        # ``setStreamValues(data)`` expects the full state dict directly.
        # Extract ``__interrupt__`` and remap it to SDK-compatible format.
        payload = dict(data) if isinstance(data, dict) else data
        interrupts = data.get("__interrupt__", [])
        if interrupts:
            payload["interrupts"] = [
                {"value": i.value, "id": getattr(i, "id", None)} for i in interrupts
            ]
        return [_format_sse("values", payload)]

    if event_type == "updates":
        # ``onUpdateEvent(data)`` expects ``data`` as ``{nodeName: updateDict}``.
        # astream_events(v2) already provides this in the correct shape.
        return [_format_sse("updates", data)]

    if event_type == "messages":
        # SDK expects ``data = [messageChunk, metadata]`` (2-tuple array).
        # It destructures as ``const [serialized, _meta] = data`` — only the
        # first element is consumed by ``MessageTupleManager.add()``.
        # The second element (metadata) is optional; we include an empty dict
        # for full compliance with the ``MessagesStreamPart`` type signature.
        msg, metadata = data
        msg_type, msg_class = classify_message_type(msg)
        chunk: dict[str, Any] = {
            "content": msg.content if hasattr(msg, "content") else str(msg),
            "type": msg.type if hasattr(msg, "type") else "unknown",
            "msg_class": msg_class,
            "msg_type": msg_type,
        }
        if hasattr(msg, "id") and msg.id:
            chunk["id"] = msg.id
        # Full 2-tuple format: [messageDict, metadataDict]
        return [_format_sse("messages", [chunk, {}])]

    if event_type == "end":
        return [_format_sse("end", {"status": "completed"})]

    return []


# ── Stream History ───────────────────────────────────────────────────────────


async def _stream_run_history(
    graph: CompiledGraph,
    config: RunnableConfig,
    thread_id: str,
    run_id: str,
) -> AsyncGenerator[dict, None]:
    """Stream the full history of a completed run as SSE events."""
    yield _format_sse("metadata", {"run_id": run_id, "thread_id": thread_id})
    try:
        async for state in graph.aget_state_history(config, limit=100):
            checkpoint = getattr(state, "checkpoint", {}) or {}
            # Send state values directly (Chat UI expects top-level messages key)
            values = (
                dict(state.values) if isinstance(state.values, dict) else state.values
            )
            if isinstance(values, dict) and "checkpoint" not in values:
                values["checkpoint"] = {
                    "checkpoint_id": checkpoint.get("checkpoint_id", ""),
                    "parent_checkpoint_id": checkpoint.get("parent_checkpoint_id"),
                }
            yield _format_sse("values", values)
    except Exception:
        logger.exception("stream_run_history failed")
        yield _format_sse("error", {"message": "历史回放失败"})
    yield _format_sse("end", {"status": "completed"})


# ── Run Endpoint Implementation ──────────────────────────────────────────────


async def _resolve_input_data(
    graph: CompiledGraph,
    config: RunnableConfig,
    input_data: dict,
    thread_id: str,
) -> dict | Command | None:
    """Resolve input data for a run: detect interrupts, resume, or restart."""
    try:
        current_state = await graph.aget_state(config)
        if current_state.next:
            has_interrupt = any(task.interrupts for task in (current_state.tasks or []))
            if has_interrupt:
                resume_value = input_data.get("resume", input_data)
                logger.info("[run] Resuming interrupted thread=%s", thread_id)
                return Command(resume=resume_value)
            else:
                logger.info("[run] Continuing thread=%s from checkpoint", thread_id)
                return None

        if input_data:
            logger.info("[run] Starting new run on thread=%s", thread_id)
            return input_data
        return None
    except (ValueError, OSError, RuntimeError, TimeoutError, ConnectionError):
        return input_data or None


async def _create_run_impl(thread_id: str, run: RunRequest) -> Response:
    """Shared implementation for creating a run on a thread."""
    # Lazy imports to avoid circular dependency with app.py
    from novelfactory.config.constants import RECURSION_LIMIT
    from novelfactory.server.app import _run_store, get_app

    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }

    # Build Runtime context from request/environment
    context: NovelContext = {
        "thread_id": thread_id,
        "user_id": getattr(run, "user_id", ""),
        "project_id": getattr(run, "project_id", ""),
        "request_id": str(uuid.uuid4()),
        "lark_config": getattr(run, "lark_config", None),
    }

    run_id = str(uuid.uuid4())
    graph = await get_app()
    input_data = await _resolve_input_data(graph, config, run.input or {}, thread_id)

    # Track the run
    run_info = {"run_id": run_id, "thread_id": thread_id, "status": "pending"}
    _run_store.setdefault(thread_id, []).append(run_info)

    if run.stream:
        return EventSourceResponse(
            _stream_run(graph, input_data, config, run_id, thread_id, context),
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming: invoke and return
    try:
        result = await graph.ainvoke(input_data, config=config, context=context)
    except (ValueError, OSError) as e:
        logger.exception("[run] invoke failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    else:
        run_info["status"] = "completed"
        from novelfactory.server.app import CustomJSONResponse

        return CustomJSONResponse(
            content={
                "run_id": run_id,
                "thread_id": thread_id,
                "status": "completed",
                "result": result,
            }
        )


async def _stream_run(
    graph: CompiledGraph,
    input_data: dict | Command,
    config: RunnableConfig,
    run_id: str,
    thread_id: str,
    context: NovelContext | None = None,
) -> AsyncGenerator[dict, None]:
    """Stream graph execution events via SSE."""
    yield _format_sse("metadata", {"run_id": run_id, "thread_id": thread_id})
    tracker = StreamStateTracker()

    try:
        async for event in graph.astream_events(
            input_data,
            config=config,
            context=context,
            version="v2",
        ):
            for sse_msg in _process_stream_event(event):
                if sse_msg.get("event") == "messages":
                    data_obj = json.loads(sse_msg["data"])
                    msg_list = data_obj if isinstance(data_obj, list) else []
                    if msg_list and isinstance(msg_list[0], dict):
                        msg_id = msg_list[0].get("id", "")
                        if tracker.is_message_duplicate(msg_id):
                            continue
                yield sse_msg

            if event.get("type") == "updates":
                data = event.get("data", {})
                if isinstance(data, dict):
                    for node_name, node_update in data.items():
                        if isinstance(node_update, dict) and tracker.update_from_state(
                            node_update
                        ):
                            yield _format_sse("progress", tracker.to_progress_event())

        for sse_msg in await _detect_stream_interrupts(graph, config, thread_id):
            yield sse_msg

    except (ValueError, OSError, RuntimeError, TimeoutError) as e:
        logger.exception("[stream] Run failed")
        yield _format_sse("error", {"message": str(e)})
    except BaseExceptionGroup as eg:
        # astream_events internally uses anyio task groups; unhandled
        # subtask exceptions are wrapped in ExceptionGroup (Python 3.11+).
        # Unwrap the first meaningful message for the client.
        exc_messages = [
            str(ex) for ex in eg.exceptions if isinstance(ex, BaseException)
        ]
        msg = exc_messages[0] if exc_messages else "任务组内部异常"
        logger.exception("[stream] TaskGroup exception: %s", msg)
        yield _format_sse("error", {"message": msg})
