"""Shared utilities for conversational sub-agents."""

from __future__ import annotations


def get_last_user_message(messages: list) -> str | None:
    """Get the last user message content from a message list."""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            return str(msg.content) if msg.content else None
        if isinstance(msg, dict) and msg.get("type") == "human":
            content = msg.get("content", "")
            return str(content) if content else None
    return None
