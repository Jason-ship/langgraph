"""Token usage statistics API — per-thread token consumption tracking."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from langchain_core.runnables import RunnableConfig

from novelfactory.server.deps import get_graph

logger = logging.getLogger(__name__)
router = APIRouter(tags=["threads"])


@router.get("/threads/{thread_id}/token-usage")
async def get_thread_token_usage(thread_id: str) -> dict:
    """Get token usage statistics for a thread.

    Returns:
        - thread_id: the thread identifier
        - total_tokens / total_input_tokens / total_output_tokens / total_runs
        - by_model: {model_name: {tokens, runs}}
        - by_caller: {lead_agent, subagent, middleware}
    """
    graph = await get_graph()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    try:
        state = await graph.aget_state(config)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found") from e

    values = state.values
    total_usage = values.get("total_usage", {}) or {}
    messages = list(values.get("messages", []) or [])

    # --- Total tokens ---
    total_input = total_usage.get("prompt_tokens", 0)
    total_output = total_usage.get("completion_tokens", 0)
    total_tokens = total_usage.get("total_tokens", total_input + total_output)

    # --- Total runs (one per message with a usage payload) ---
    total_runs = sum(1 for m in messages if _get_usage_from_message(m) is not None)

    # --- By model: {model_name: {tokens, runs}} ---
    raw_model_breakdown: dict = total_usage.get("model_breakdown", {})
    by_model: dict[str, dict[str, int]] = {}
    for model_name, model_info in raw_model_breakdown.items():
        if isinstance(model_info, dict):
            tokens = model_info.get("tokens", 0) or (
                model_info.get("input_tokens", 0) + model_info.get("output_tokens", 0)
            )
            runs = model_info.get("runs", 1)
        else:
            # Fallback: treat the value as a plain token count
            tokens = int(model_info) if model_info else 0
            runs = 1
        by_model[model_name] = {"tokens": tokens, "runs": runs}

    # --- By caller: classify into lead_agent / subagent / middleware ---
    by_caller_raw: dict[str, dict[str, int]] = {}
    for msg in messages:
        caller = getattr(msg, "name", None) or "unknown"
        usage = _get_usage_from_message(msg)
        if usage:
            if caller not in by_caller_raw:
                by_caller_raw[caller] = {"prompt_tokens": 0, "completion_tokens": 0}
            by_caller_raw[caller]["prompt_tokens"] += usage.get("input_tokens", 0)
            by_caller_raw[caller]["completion_tokens"] += usage.get("output_tokens", 0)

    by_caller = _classify_callers(by_caller_raw)

    return {
        "thread_id": thread_id,
        "total_tokens": total_tokens,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_runs": total_runs,
        "by_model": by_model,
        "by_caller": by_caller,
    }


def _get_usage_from_message(msg: Any) -> dict[str, int] | None:
    """Extract usage metadata from a message."""
    try:
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            return {
                "input_tokens": getattr(msg.usage_metadata, "input_tokens", 0),
                "output_tokens": getattr(msg.usage_metadata, "output_tokens", 0),
            }
        if hasattr(msg, "additional_kwargs"):
            return msg.additional_kwargs.get("usage_metadata")
    except Exception:
        pass
    return None


# --- Caller classification ---
# Matches project agent names from graph/crews/ and agents/infra/.

_LEAD_AGENT_PATTERNS: frozenset[str] = frozenset({
    "main_supervisor", "supervisor",
})

_SUBAGENT_PATTERNS: frozenset[str] = frozenset({
    "chapter_writer", "chapter_planner", "chapter_refiner",
    "context_builder", "state_extractor", "database_writer",
    "quality_reviewer", "ai_style_reviewer", "old_reader_reviewer",
    "debate_reviewer", "illustrator", "tts_agent",
    "setup_crew", "writing_crew", "media_crew", "sync_crew",
    "foreshadowing", "volume_check", "quality_check",
    "prepare_writing", "intelligent_monitor",
})

_MIDDLEWARE_PATTERNS: frozenset[str] = frozenset({
    "cache", "middleware", "retry", "circuit_breaker",
})


def _classify_callers(
    by_caller_raw: dict[str, dict[str, int]],
) -> dict[str, int]:
    """Classify per-caller token counts into lead_agent / subagent / middleware.

    Tokens from callers that match known project agent names are bucketed
    accordingly.  Unknown callers default to *subagent* (the most common
    category).  Each bucket sums prompt + completion tokens.
    """
    totals: dict[str, int] = {"lead_agent": 0, "subagent": 0, "middleware": 0}

    for caller, usage in by_caller_raw.items():
        caller_lower = caller.lower()
        combined = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

        if caller_lower in _LEAD_AGENT_PATTERNS:
            totals["lead_agent"] += combined
        elif caller_lower in _MIDDLEWARE_PATTERNS:
            totals["middleware"] += combined
        else:
            totals["subagent"] += combined

    return totals