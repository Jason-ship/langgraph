"""State access helpers and AI message text extraction."""

from __future__ import annotations

from typing import Any


def make_retry_agent_invoke(module_name: str):
    """Create a module-specific ``_retry_agent_invoke`` function.

    Factory pattern eliminates duplication of the retry wrapper between
    writing_agents.py and setup_agents.py. Each module calls this factory
    once at import time, getting a closure that hardcodes the step_name
    prefix — so all existing ``_retry_agent_invoke(agent, input, step)``
    call sites remain unchanged.

    Args:
        module_name: Prefix for step_name (e.g. "writing_agents", "setup_agents").

    Returns:
        A ``_retry_agent_invoke(agent, input_dict, step_name) -> dict`` function.
    """

    def _retry_agent_invoke(agent: Any, input_dict: dict, step_name: str) -> dict:
        """Production-grade agent.invoke with timeout + exponential-backoff retry.

        Falls back to ``{"messages": [], "crew_result": {}}`` on all failures.
        """
        from novelfactory.agents.infra.retry import llm_call_with_retry

        return llm_call_with_retry(
            agent.invoke,
            input_dict,
            step_name=f"{module_name}.{step_name}",
            fallback={"messages": [], "crew_result": {}},
        )

    return _retry_agent_invoke


def _extract_from_state(state: dict, key: str, default: Any = "") -> Any:
    """Read `key` from `state`, supporting both flat and crew_result layouts."""
    if "crew_result" in state and isinstance(state.get("crew_result"), dict):
        return state["crew_result"].get(key, default)
    return state.get(key, default)


def extract_fields_from_state(state: dict, fields: dict[str, Any]) -> dict[str, Any]:
    """Batch extract multiple fields from state, supporting crew_result/flat layouts.

    v6.1 P2-1: Unified replacement for the 5 duplicate _get_context functions.
    Each agent calls this with its own field mapping, eliminating boilerplate
    while preserving agent-specific field selection.

    Args:
        state: The state dict (may contain "crew_result" nested dict).
        fields: Mapping of {field_name: default_value} to extract.

    Returns:
        Dict of {field_name: value} extracted from crew_result or top-level.
    """
    source = (
        state.get("crew_result", state)
        if isinstance(state.get("crew_result"), dict)
        else state
    )
    # Fallback: if crew_result exists but field is missing, try top-level
    result = {}
    for key, default in fields.items():
        val = source.get(key, ...)
        if val is ...:
            val = state.get(key, default)
        result[key] = val
    return result


# ── Public Utility: extract_ai_message_text ─────────────────────────────────
# v5.1.1: 从 5 个 agent 文件 (setup/writing/review/media/sync) 中提取公共实现。
# 原 _to_messages() 在 5 个文件中各有相同逻辑（仅 write 版缺少 .strip()），
# 现统一为带 strip_whitespace 参数的公共函数。


def extract_ai_message_text(result: dict, *, strip_whitespace: bool = True) -> str:
    """从 create_react_agent 结果中提取最后一条 AI 消息的文本内容。

    遍历 messages 列表（倒序查找最后一条 AI 消息），处理两种消息格式：
      - langchain Message 对象 (hasattr msg.type == "ai")
      - 纯 dict 消息 (msg["type"] == "ai")

    content 可能为 str 或 list[dict] (多模态 content parts)，
    对 list 类型拼接所有 type=="text" 的 part.text。

    Args:
        result: create_react_agent.invoke() 的返回 dict，包含 "messages" 键。
        strip_whitespace: 是否对返回文本调用 .strip()。默认 True。

    Returns:
        提取到的文本内容，未找到消息时返回空字符串。
    """
    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, "type") and msg.type == "ai":
            content = msg.content
        elif isinstance(msg, dict) and msg.get("type") == "ai":
            content = msg.get("content", "")
        else:
            continue
        if isinstance(content, str):
            return content.strip() if strip_whitespace else content
        if isinstance(content, list):
            text = "".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
            return text.strip() if strip_whitespace else text
    return ""
