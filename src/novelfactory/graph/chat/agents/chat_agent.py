"""General chat sub-agent — handles non-writing conversations."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from novelfactory.config.llm import get_worker_llm
from novelfactory.graph.chat.agents.utils import get_last_user_message
from novelfactory.state.lead_agent_state import LeadAgentState

logger = logging.getLogger(__name__)


async def chat_agent_node(state: LeadAgentState) -> dict[str, Any]:
    """General chat node — handles non-writing conversations.

    This is the fallback agent when the intent doesn't match any specific
    writing-related sub-agent. It handles:
    - General questions about the novel writing process
    - Help and guidance requests
    - Casual conversation
    """
    llm = get_worker_llm()
    messages = state.get("messages", [])

    # Get the last user message
    last_user_msg = get_last_user_message(messages)
    if not last_user_msg:
        return {
            "current_agent": "chat_agent",
            "messages": [
                AIMessage(
                    content="你好！我是 NovelFactory 创作助手。我可以帮你：\n\n"
                    "1. 📖 **故事策划** — 构建世界观、角色和情节\n"
                    "2. ✍️ **章节写作** — 生成小说章节内容\n"
                    "3. 🔍 **质量评审** — 检查章节质量\n"
                    "4. 💬 **一般问答** — 回答创作相关问题\n\n"
                    "你想做什么？",
                    name="chat_agent",
                )
            ],
        }

    # Build context for the LLM
    context = _build_chat_context(state)

    prompt = (
        "你是一位友好的小说创作助手，帮助用户完成小说创作。\n\n"
        f"## 当前项目状态\n{context}\n\n"
        f"## 用户消息\n{last_user_msg}\n\n"
        "## 回答原则\n"
        "1. 回答要亲切、专业、鼓励\n"
        "2. 如果用户想创作小说，引导他们使用故事策划、章节写作或评审功能\n"
        "3. 对于创作技巧类问题，给出具体、可操作的建议\n"
        "4. 保持回答简洁（不超过300字）"
    )

    try:
        response = await llm.ainvoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        return {
            "current_agent": "chat_agent",
            "messages": [AIMessage(content=content, name="chat_agent")],
        }
    except Exception as e:
        logger.exception("[ChatAgent] Chat failed: %s", e)
        return {
            "current_agent": "chat_agent",
            "messages": [
                AIMessage(
                    content=f"处理消息时遇到问题: {e}，请重试。",
                    name="chat_agent",
                )
            ],
        }


def _build_chat_context(state: LeadAgentState) -> str:
    """Build context from state for the chat prompt."""
    parts = []
    if state.get("project_name"):
        parts.append(f"项目: {state['project_name']}")
    if state.get("genre"):
        parts.append(f"题材: {state['genre']}")
    if state.get("current_chapter"):
        parts.append(f"进度: 第{state['current_chapter']}章")
    if state.get("context_summary"):
        parts.append(f"对话摘要: {state['context_summary']}")
    if not parts:
        parts.append("新对话，尚未开始创作")
    return "\n".join(parts)