"""Story planning sub-agent — handles world-building, character design, outlines."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from novelfactory.config.llm import get_worker_llm
from novelfactory.graph.chat.agents.utils import get_last_user_message
from novelfactory.state.lead_agent_state import LeadAgentState

logger = logging.getLogger(__name__)


async def story_agent_node(state: LeadAgentState) -> dict[str, Any]:
    """Story planning node — processes story-related requests.

    Handles:
    - World building conversations
    - Character design discussions
    - Story outline planning
    - Genre and theme exploration
    """
    llm = get_worker_llm()
    messages = state.get("messages", [])

    last_user_msg = get_last_user_message(messages)
    if not last_user_msg:
        return {
            "current_agent": "story_agent",
            "messages": [
                AIMessage(
                    content="你好！我是你的故事策划师。我们可以一起构建精彩的故事世界！\n\n"
                    "你想从哪里开始呢？\n"
                    "1. 📖 **故事题材** — 确定小说的类型和风格\n"
                    "2. 🌍 **世界观构建** — 创造独特的故事世界\n"
                    "3. 👤 **角色设计** — 塑造鲜活的人物\n"
                    "4. 📋 **大纲规划** — 设计情节结构",
                    name="story_agent",
                )
            ],
        }

    # Determine if this is a structured request or free-form conversation
    context = _build_story_context(state)

    prompt = (
        "你是一位专业的故事策划师，擅长帮助作者构建小说世界观、角色和情节。\n\n"
        f"## 当前创作状态\n{context}\n\n"
        f"## 用户消息\n{last_user_msg}\n\n"
        "## 你的角色\n"
        "- 通过提问引导用户逐步完善故事设定\n"
        "- 提供专业建议和创意启发\n"
        "- 记录用户重要的创作决策\n\n"
        "## 输出要求\n"
        "1. 回应要亲切、鼓励\n"
        "2. 每次聚焦一个话题\n"
        "3. 适时总结已经确定的设定\n"
        "4. 提出有启发性的问题引导用户思考"
    )

    try:
        response = await llm.ainvoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        # Extract structured info from the conversation if available
        updates = _extract_story_updates(content, state)

        return {
            "current_agent": "story_agent",
            "messages": [AIMessage(content=content, name="story_agent")],
            **updates,
        }
    except Exception as e:
        logger.exception("[StoryAgent] Planning failed: %s", e)
        return {
            "current_agent": "story_agent",
            "messages": [AIMessage(content=f"处理时遇到问题: {e}，请重试。", name="story_agent")],
        }


def _build_story_context(state: LeadAgentState) -> str:
    """Build context from existing story state."""
    parts = []
    if state.get("genre"):
        parts.append(f"题材: {state['genre']}")
    if state.get("world_setting"):
        parts.append(f"世界观: {state['world_setting'][:300]}")
    if state.get("character_setting"):
        parts.append(f"角色: {state['character_setting'][:300]}")
    if state.get("story_outline"):
        parts.append(f"大纲: {state['story_outline'][:300]}")
    if state.get("setup_complete"):
        parts.append("✅ 基础设定已完成")
    if state.get("context_summary"):
        parts.append(f"对话摘要: {state['context_summary']}")
    return "\n".join(parts) if parts else "全新的故事项目，尚未有任何设定"


def _extract_story_updates(content: str, state: LeadAgentState) -> dict[str, Any]:
    """Extract structured story information from LLM response.

    This is a simple heuristic-based extraction. In production, this would
    use a structured output parser.
    """
    updates = {}
    content_lower = content.lower()

    # Detect genre mentions
    genres = {
        "玄幻": "xianxia",
        "修仙": "xianxia",
        "修真": "xianxia",
        "科幻": "sci-fi",
        "未来": "sci-fi",
        "都市": "urban",
        "现代": "urban",
        "历史": "historical",
        "古代": "historical",
        "悬疑": "mystery",
        "推理": "mystery",
        "恐怖": "horror",
        "惊悚": "horror",
        "言情": "romance",
        "爱情": "romance",
        "游戏": "game",
        "网游": "game",
    }
    for keyword, genre in genres.items():
        if keyword in content_lower and not state.get("genre"):
            updates["genre"] = genre
            break

    return updates