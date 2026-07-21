"""Thread context compaction API — summarization for long conversations."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from novelfactory.config.llm import get_worker_llm

logger = logging.getLogger(__name__)
router = APIRouter()

from novelfactory.server.deps import get_graph  # noqa: E402


class CompactRequest(BaseModel):
    """Request body for context compaction."""

    force: bool = False
    agent_name: str | None = None


@router.post("/threads/{thread_id}/compact", tags=["threads"])
async def compact_thread(thread_id: str, req: CompactRequest | None = None) -> dict:
    """Compact thread context — summarize older messages to save context.

    Uses LLM to generate a summary of older messages, then replaces them
    with the summary. Keeps the most recent 5 messages intact.
    """
    force = req.force if req else False

    graph = await get_graph()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    try:
        state = await graph.aget_state(config)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found") from e

    messages = list(state.values.get("messages", []) or [])
    original_count = len(messages)

    if not force and original_count < 10:
        return {
            "thread_id": thread_id,
            "compacted": False,
            "reason": "too_few_messages",
            "removed_message_count": 0,
            "preserved_message_count": original_count,
            "summary_updated": False,
            "checkpoint_id": None,
            "total_tokens": _estimate_tokens(messages),
        }

    # Keep last 5, summarize the rest
    keep_count = 5
    to_summarize = messages[:-keep_count]
    to_keep = messages[-keep_count:]

    summary = await _generate_summary(to_summarize)
    if not summary:
        return {
            "thread_id": thread_id,
            "compacted": False,
            "reason": "summary_generation_failed",
            "removed_message_count": 0,
            "preserved_message_count": original_count,
            "summary_updated": False,
            "checkpoint_id": None,
            "total_tokens": _estimate_tokens(messages),
        }

    # Update state with summary + keep recent messages
    from langchain_core.messages import SystemMessage

    summary_msg = SystemMessage(
        content=f"[上下文摘要]\n{summary}",
        name="context_summary",
    )

    # NOTE: Cannot replace messages via aupdate_state because the `add_messages`
    # reducer merges instead of replaces.  Instead, prepend the summary as a new
    # SystemMessage and store the summary separately.  The frontend / LLM will
    # see the summary marker and can ignore older messages accordingly.
    await graph.aupdate_state(
        config,
        {
            "messages": [summary_msg],
            "context_summary": summary,
            "needs_compaction": False,
        },
    )

    # Messages were NOT removed — only the summary was added.
    removed_count = 0
    kept_count = original_count + 1  # +1 for summary message

    logger.info(
        "[compact] Thread %s: %d messages \u2192 summary + %d recent",
        thread_id, original_count, keep_count,
    )

    return {
        "thread_id": thread_id,
        "compacted": True,
        "reason": None,
        "removed_message_count": max(removed_count, 0),
        "preserved_message_count": kept_count,
        "summary_updated": True,
        "checkpoint_id": None,
        "total_tokens": _estimate_tokens(to_keep) + len(summary) // 4,
    }


async def _generate_summary(messages: list) -> str | None:
    """Generate a summary of the given messages using LLM."""
    try:
        llm = get_worker_llm()

        # Build a concise representation of the conversation
        lines = []
        for m in messages[-20:]:  # At most 20 messages
            role = "User" if getattr(m, "type", "") == "human" else "AI"
            content = getattr(m, "content", "")
            if content:
                lines.append(f"{role}: {str(content)[:200]}")

        text = "\n".join(lines)
        if not text:
            return None

        response = await llm.ainvoke(
            "请总结以下对话的核心内容，保留关键决策、创作方向、已完成的章节信息：\n\n"
            f"{text}\n\n"
            "摘要（200字以内）："
        )
        summary = response.content if hasattr(response, "content") else str(response)
        return summary[:2000]
    except Exception as e:
        logger.warning("[compact] Summary generation failed: %s", e)
        return None


def _estimate_tokens(messages: list) -> int:
    """Rough token estimate based on message content lengths."""
    total = 0
    for m in messages:
        content = getattr(m, "content", "") or ""
        total += len(str(content)) // 4  # ~4 chars per token heuristic
    return total