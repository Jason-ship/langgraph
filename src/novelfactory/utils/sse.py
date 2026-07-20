"""SSE 格式化工具 — 与 LangGraph Platform 线格式兼容。

Migrated from DeerFlow app/gateway/services.py.
"""

from __future__ import annotations

from typing import Any


def format_sse(event: str, data: Any, event_id: str | None = None) -> str:
    """格式化 SSE 帧。

    字段顺序: event: → data: → id: → 空行
    匹配 LangGraph Platform 线格式。

    Args:
        event: 事件类型。
        data: 事件数据（会被 JSON 序列化）。
        event_id: 可选事件 ID。

    Returns:
        格式化后的 SSE 字符串。
    """
    import json

    lines = [f"event: {event}", f"data: {json.dumps(data, ensure_ascii=False, default=str)}"]
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append("")
    return "\n".join(lines)


def format_sse_event(event: str, data: Any) -> str:
    """简化版 SSE 格式化。"""
    return format_sse(event, data)


__all__ = [
    "format_sse",
    "format_sse_event",
]