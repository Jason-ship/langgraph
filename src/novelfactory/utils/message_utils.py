# ==============================================================================
# 消息分类与去重工具 — 借鉴 TradingAgents cli/utils.py classify_message_type 模式
# ==============================================================================

from __future__ import annotations

from typing import Any


def classify_message_type(msg: Any) -> tuple[str, str]:
    """将 LangChain 消息分类为显示类型。

    借鉴 TradingAgents: 消息按角色分类 → User/Agent/Data/System/Control
    """
    msg_type = (
        getattr(msg, "type", "") if hasattr(msg, "type") else str(type(msg).__name__)
    )

    mapping = {
        "human": ("Human", "User"),
        "ai": ("AI", "Agent"),
        "tool": ("Tool", "Data"),
        "system": ("System", "System"),
    }
    return mapping.get(msg_type, (msg_type, "Info"))


def deduplicate_messages(messages: list[dict], *, key_fn: Any = None) -> list[dict]:
    """按 id 去重消息列表，保留最后出现的。"""
    seen: set[str] = set()
    result: list[dict] = []
    for msg in reversed(messages):
        msg_id = msg.get("id", "")
        if key_fn:
            msg_id = key_fn(msg)
        if msg_id and msg_id in seen:
            continue
        if msg_id:
            seen.add(msg_id)
        result.append(msg)
    result.reverse()
    return result
