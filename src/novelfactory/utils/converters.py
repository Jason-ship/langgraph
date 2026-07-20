"""消息转换器 — LangChain 消息 ↔ OpenAI 格式互转。

参考 DeerFlow runtime/converters.py 的纯函数模式。

适用于需要与 OpenAI 兼容 API 交互的场景。
"""

from __future__ import annotations

import json
from typing import Any

_ROLE_MAP = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
}


def _infer_finish_reason(message: Any) -> str:
    """从 response_metadata 推断 finish_reason。"""
    metadata = getattr(message, "response_metadata", None) or {}
    if isinstance(metadata, dict):
        reason = metadata.get("finish_reason")
        if isinstance(reason, str) and reason:
            return reason
    return "stop"


def langchain_to_openai_message(message: Any) -> dict[str, Any]:
    """将单个 LangChain BaseMessage 转换为 OpenAI 消息 dict。

    Args:
        message: LangChain BaseMessage 实例。

    Returns:
        OpenAI 格式的消息 dict。
    """
    msg_type = getattr(message, "type", "")
    role = _ROLE_MAP.get(msg_type, msg_type)
    content = getattr(message, "content", "")

    if role == "tool":
        return {
            "role": "tool",
            "tool_call_id": getattr(message, "tool_call_id", ""),
            "content": content,
        }

    if role == "assistant":
        tool_calls = getattr(message, "tool_calls", None) or []
        result: dict[str, Any] = {"role": "assistant"}

        if isinstance(content, str) and content:
            result["content"] = content
        elif isinstance(content, list):
            result["content"] = content
        else:
            result["content"] = ""

        if tool_calls:
            openai_tool_calls = []
            for tc in tool_calls:
                args = tc.get("args", {})
                openai_tool_calls.append({
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(args) if not isinstance(args, str) else args,
                    },
                })
            result["tool_calls"] = openai_tool_calls

        return result

    # user / system
    return {"role": role, "content": content}


def langchain_to_openai_completion(message: Any, model: str = "") -> dict[str, Any]:
    """将 AIMessage 转换为完整的 OpenAI 补全响应格式。

    Args:
        message: AIMessage 实例。
        model: 模型名称。

    Returns:
        OpenAI 补全格式的响应 dict。
    """
    import uuid

    content = message.content if isinstance(message.content, str) else ""
    finish_reason = _infer_finish_reason(message)

    usage = getattr(message, "response_metadata", {}).get("token_usage", {}) if hasattr(message, "response_metadata") else {}

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "model": model or "unknown",
        "choices": [
            {
                "index": 0,
                "message": langchain_to_openai_message(message),
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0,
            "completion_tokens": usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0,
            "total_tokens": usage.get("total_tokens", 0) if isinstance(usage, dict) else 0,
        },
    }


__all__ = [
    "langchain_to_openai_message",
    "langchain_to_openai_completion",
]