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


__all__ = [
    "message_content_to_text",
    "message_to_text",
    "ORIGINAL_USER_CONTENT_KEY",
]