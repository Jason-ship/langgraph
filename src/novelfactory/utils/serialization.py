"""Canonical serialization for LangChain / LangGraph objects.

Migrated from DeerFlow runtime/serialization.py.

Provides a single source of truth for converting LangChain message
objects, Pydantic models, and LangGraph state dicts into plain
JSON-serialisable Python structures.
"""

from __future__ import annotations

from typing import Any


def serialize_lc_object(obj: Any) -> Any:
    """Recursively serialize a LangChain object to a JSON-serialisable dict."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: serialize_lc_object(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_lc_object(item) for item in obj]
    # Pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # Pydantic v1
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    # LangGraph Interrupt
    try:
        from langgraph.types import Interrupt

        if isinstance(obj, Interrupt):
            return serialize_lc_object({"value": obj.value, "id": getattr(obj, "id", None)})
    except ImportError:
        pass
    # Last resort
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def serialize_channel_values(channel_values: dict[str, Any]) -> dict[str, Any]:
    """Serialize channel values, stripping internal LangGraph keys."""
    result = {}
    for key, value in channel_values.items():
        if key.startswith("__"):
            continue
        result[key] = serialize_lc_object(value)
    return result


def serialize_interrupts(interrupts: list[Any]) -> list[dict[str, Any]]:
    """Serialize LangGraph Interrupt objects."""
    return [serialize_lc_object(interrupt) for interrupt in interrupts]


__all__ = [
    "serialize_lc_object",
    "serialize_channel_values",
    "serialize_interrupts",
]