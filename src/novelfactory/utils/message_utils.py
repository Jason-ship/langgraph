"""Message content extraction and conversion utilities.

Migrated from DeerFlow utils/messages.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

ORIGINAL_USER_CONTENT_KEY = "original_user_content"


def message_content_to_text(content: Any) -> str:
    """Extract text from LangChain message content shapes.

    Handles plain strings, lists of content blocks, and nested dicts.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(content)


def message_to_text(message: Any, *, text_attribute_fallback: bool = False) -> str:
    """Extract display text from a whole message (BaseMessage or dict-shaped).

    Reads content from either an attribute (BaseMessage) or a mapping key,
    then walks the mixed content shapes.
    """
    content = message.get("content") if isinstance(message, Mapping) else getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    nested = block.get("content")
                    if isinstance(nested, str):
                        parts.append(nested)
        return "".join(parts)
    if isinstance(content, Mapping):
        for key in ("text", "content"):
            value = content.get(key)
            if isinstance(value, str):
                return value
    if text_attribute_fallback:
        text = getattr(message, "text", None)
        if isinstance(text, str):
            return text
    return ""


def classify_message_type(msg: Any) -> tuple[str, str]:
    """Classify a message into (msg_type, msg_class) for SSE streaming.

    Args:
        msg: A LangChain BaseMessage or dict-like message object.

    Returns:
        Tuple of (msg_type, msg_class).
        msg_type: "ai", "human", "tool", "system", "unknown"
        msg_class: Short description of the message content/subtype.
    """
    # Determine type
    if hasattr(msg, "type"):
        raw_type = msg.type
    elif isinstance(msg, dict):
        raw_type = msg.get("type", "unknown")
    else:
        raw_type = "unknown"

    msg_type = raw_type if raw_type in ("ai", "human", "tool", "system") else "unknown"

    # Determine class
    content = ""
    if hasattr(msg, "content"):
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
    elif isinstance(msg, dict):
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)

    if msg_type == "ai":
        if hasattr(msg, "name") and msg.name:
            msg_class = msg.name
        elif isinstance(msg, dict) and msg.get("name"):
            msg_class = msg["name"]
        else:
            msg_class = "ai_assistant"
    elif msg_type == "tool":
        msg_class = "tool_call"
    elif msg_type == "human":
        msg_class = "user"
    elif msg_type == "system":
        msg_class = "system"
    else:
        msg_class = "unknown"

    return msg_type, msg_class


__all__ = [
    "message_content_to_text",
    "message_to_text",
    "classify_message_type",
    "ORIGINAL_USER_CONTENT_KEY",
]