"""Thread context compaction — summarize old messages to keep context within budget.

Migrated from DeerFlow runtime/context_compaction.py.

NovelFactory's long-running chapter creation threads accumulate messages over
many turns. This module provides manual compaction: summarize old messages
and write a compacted checkpoint, preserving only the recent active window.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class ContextCompactionDisabled(RuntimeError):
    """Raised when compaction is requested but summarization is not configured."""


class ContextCompactionFailed(RuntimeError):
    """Raised when a thread cannot be compacted."""


@dataclass(frozen=True)
class ThreadCompactionResult:
    """Result returned after a context-compaction attempt.

    Attributes:
        thread_id: The thread that was compacted.
        compacted: Whether compaction actually occurred.
        reason: If not compacted, why.
        removed_message_count: Number of messages summarized away.
        preserved_message_count: Number of messages kept.
        summary_updated: Whether the summary was updated.
        checkpoint_id: The new checkpoint ID (if compacted).
        total_tokens: Token count of the generated summary.
    """

    thread_id: str
    compacted: bool
    reason: str | None = None
    removed_message_count: int = 0
    preserved_message_count: int = 0
    summary_updated: bool = False
    checkpoint_id: str | None = None
    total_tokens: int = 0


def _estimate_messages_token_count(messages: list[dict]) -> int:
    """Roughly estimate the token count of a list of messages."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content) // 2  # rough: ~2 chars per token for CJK
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += len(block.get("text", "")) // 2
    return total


def _build_summary_from_messages(messages: list[dict], max_tokens: int = 2000) -> str:
    """Build a text summary from a list of messages.

    Since we don't have an LLM call here, we extract key information:
    - Tool call results
    - AI assistant responses
    - Human inputs (truncated)
    """
    parts: list[str] = []
    token_budget = max_tokens

    for msg in messages:
        msg_type = msg.get("type", "")
        content = msg.get("content", "")
        name = msg.get("name", "")

        if not isinstance(content, str) or not content:
            continue

        # Estimate tokens for this message
        msg_tokens = len(content) // 2
        if msg_tokens > token_budget:
            content = content[: token_budget * 2] + "..."

        if msg_type == "human":
            parts.append(f"User: {content[:200]}")
        elif msg_type == "ai":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                call_names = [tc.get("name", "?") for tc in tool_calls if isinstance(tc, dict)]
                parts.append(f"Assistant: [tool_calls: {', '.join(call_names)}]")
            else:
                parts.append(f"Assistant: {content[:300]}")
        elif msg_type == "tool":
            parts.append(f"Tool ({name}): {content[:200]}")

        token_budget -= msg_tokens
        if token_budget <= 0:
            break

    summary = "\n".join(parts)
    return summary[: max_tokens * 2] if len(summary) > max_tokens * 2 else summary


async def compact_thread_context(
    checkpointer: Any,
    thread_id: str,
    *,
    keep_last_n: int = 20,
    min_messages: int = 50,
    force: bool = False,
    llm: Any = None,
) -> ThreadCompactionResult:
    """Summarize old messages in a thread and write a compacted checkpoint.

    Args:
        checkpointer: The checkpointer instance (must have aget_tuple/put methods).
        thread_id: The thread to compact.
        keep_last_n: Number of recent messages to preserve in full.
        min_messages: Minimum messages required before compaction triggers.
        force: If True, compact even if below min_messages.
        llm: Optional LLM for generating a better summary. If None, uses heuristic.

    Returns:
        ThreadCompactionResult with compaction details.
    """
    # Read the latest checkpoint
    read_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    try:
        checkpoint_tuple = await checkpointer.aget_tuple(read_config)
    except Exception:
        checkpoint_tuple = None

    if checkpoint_tuple is None:
        return ThreadCompactionResult(thread_id=thread_id, compacted=False, reason="checkpoint_not_found")

    checkpoint: dict[str, Any] = dict(getattr(checkpoint_tuple, "checkpoint", {}) or {})
    channel_values: dict[str, Any] = dict(checkpoint.get("channel_values", {}) or {})
    messages = channel_values.get("messages", [])

    if not isinstance(messages, list) or len(messages) < min_messages:
        if not force:
            return ThreadCompactionResult(
                thread_id=thread_id,
                compacted=False,
                reason=f"not_enough_messages ({len(messages) if isinstance(messages, list) else 0} < {min_messages})",
            )

    if not isinstance(messages, list) or len(messages) <= keep_last_n:
        return ThreadCompactionResult(thread_id=thread_id, compacted=False, reason="below_keep_threshold")

    # Split: old messages to summarize, recent messages to keep
    summarize_count = len(messages) - keep_last_n
    messages_to_summarize = list(messages[:summarize_count])
    preserved_messages = list(messages[-keep_last_n:])

    # Build summary
    summary_text = _build_summary_from_messages(messages_to_summarize)
    existing_summary = channel_values.get("summary_text", "")
    if existing_summary:
        summary_text = f"{existing_summary}\n\n[Continued]\n{summary_text}"

    token_count = _estimate_messages_token_count(messages_to_summarize)

    logger.info(
        "[compact] Thread %s: summarized %d messages → %d chars, keeping %d recent messages",
        thread_id,
        summarize_count,
        len(summary_text),
        keep_last_n,
    )

    # Write compacted checkpoint
    channel_values["messages"] = preserved_messages
    channel_values["summary_text"] = summary_text
    checkpoint["channel_values"] = channel_values

    metadata: dict[str, Any] = dict(getattr(checkpoint_tuple, "metadata", {}) or {})
    metadata["source"] = "context_compaction"
    metadata["compacted_at"] = __import__("datetime").datetime.now().isoformat()
    metadata["compacted_message_count"] = summarize_count

    write_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    try:
        new_config = await checkpointer.aput(write_config, checkpoint, metadata, {})
    except Exception as exc:
        raise ContextCompactionFailed(f"Failed to write compacted checkpoint: {exc}") from exc

    new_checkpoint_id = None
    if isinstance(new_config, dict):
        new_checkpoint_id = new_config.get("configurable", {}).get("checkpoint_id")

    return ThreadCompactionResult(
        thread_id=thread_id,
        compacted=True,
        removed_message_count=summarize_count,
        preserved_message_count=len(preserved_messages),
        summary_updated=True,
        checkpoint_id=new_checkpoint_id,
        total_tokens=token_count,
    )


__all__ = [
    "ThreadCompactionResult",
    "compact_thread_context",
    "ContextCompactionDisabled",
    "ContextCompactionFailed",
]