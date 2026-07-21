"""Chapter writing sub-agent — generates chapter content via conversation."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from novelfactory.config.llm import get_worker_llm
from novelfactory.graph.chat.agents.utils import get_last_user_message
from novelfactory.state.lead_agent_state import LeadAgentState

logger = logging.getLogger(__name__)


async def writing_agent_node(state: LeadAgentState) -> dict[str, Any]:
    """Chapter writing node — processes chapter writing requests.

    Extracts the user's writing request from the conversation history,
    calls the LLM to generate chapter content, and returns the result.

    Supports:
    - Writing a new chapter from scratch
    - Continuing from an existing chapter
    - Incorporating user feedback into an existing draft
    """
    llm = get_worker_llm()
    messages = state.get("messages", [])

    # Get the last user message as the writing prompt
    last_user_msg = get_last_user_message(messages)
    if not last_user_msg:
        return {
            "current_agent": "writing_agent",
            "messages": [AIMessage(content="请告诉我你想写什么样的章节内容？", name="writing_agent")],
        }

    # Build context from previous messages
    context = _build_writing_context(state)

    # Build the writing prompt
    prompt = (
        "你是一位专业的小说章节写手。根据用户的要求创作小说章节内容。\n\n"
        f"## 当前创作上下文\n{context}\n\n"
        f"## 用户要求\n{last_user_msg}\n\n"
        "## 输出要求\n"
        "1. 生成完整的章节内容（2000-3000字）\n"
        "2. 包含章节标题\n"
        "3. 语言生动，有画面感\n"
        "4. 注意段落节奏，长短句结合\n\n"
        "请直接输出章节内容，不需要额外说明。"
    )

    try:
        logger.info("[WritingAgent] Generating chapter content...")
        response = await llm.ainvoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        return {
            "current_agent": "writing_agent",
            "chapter_draft": content,
            "messages": [
                AIMessage(
                    content=f"✅ 章节已生成（{len(content)}字）\n\n{content[:500]}...\n\n你可以要求我修改、续写，或者让评审来检查质量。",
                    name="writing_agent",
                )
            ],
            "agent_context": {"type": "chapter_writing", "word_count": len(content)},
        }
    except Exception as e:
        logger.exception("[WritingAgent] Generation failed: %s", e)
        return {
            "current_agent": "writing_agent",
            "messages": [AIMessage(content=f"生成章节时遇到问题: {e}，请重试或调整要求。", name="writing_agent")],
        }


def _build_writing_context(state: LeadAgentState) -> str:
    """Build context from existing state for the writing prompt."""
    parts = []
    if state.get("genre"):
        parts.append(f"题材: {state['genre']}")
    if state.get("project_name"):
        parts.append(f"项目: {state['project_name']}")
    if state.get("current_chapter"):
        parts.append(f"当前章节: 第{state['current_chapter']}章")
    if state.get("world_setting"):
        parts.append(f"世界观: {state['world_setting'][:200]}")
    if state.get("character_setting"):
        parts.append(f"角色设定: {state['character_setting'][:200]}")
    if state.get("story_outline"):
        parts.append(f"故事大纲: {state['story_outline'][:200]}")
    if state.get("chapter_draft"):
        parts.append(f"现有草稿: {state['chapter_draft'][:300]}...")
    if state.get("context_summary"):
        parts.append(f"对话摘要: {state['context_summary']}")
    return "\n".join(parts) if parts else "尚无创作上下文"