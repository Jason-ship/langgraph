"""Chat Supervisor — intent-based routing for the Lead Agent graph.

Analyzes user messages to determine which sub-agent should handle the request.
Supports:
- Slash commands (/write, /review, /plan, /edit)
- LLM-based intent analysis for natural language requests
- Context-aware routing (considers conversation history)
"""

from __future__ import annotations

import logging
from typing import Any

from novelfactory.config.llm import get_worker_llm

logger = logging.getLogger(__name__)

# Slash command routing table
SLASH_COMMAND_MAP: dict[str, str] = {
    "/write": "writing_agent",
    "/review": "review_agent",
    "/plan": "story_agent",
    "/story": "story_agent",
    "/edit": "writing_agent",
    "/help": "chat_agent",
    "/start": "story_agent",
}

# Intent keywords for LLM-free routing
INTENT_KEYWORDS: dict[str, list[str]] = {
    "story_agent": ["世界观", "角色", "设定", "大纲", "策划", "规划", "世界", "人物", "背景"],
    "writing_agent": ["写", "章", "节", "生成", "创作", "续写", "下一章", "内容"],
    "review_agent": ["评审", "审核", "评价", "修改", "润色", "质量", "评分", "review", "检查"],
}

# ── Batch Mode Detection ──────────────────────────────────────────────────────
# Keywords that trigger the bridge agent to delegate to the batch pipeline.
# These are checked BEFORE keyword/LLM routing so batch mode takes priority.
BATCH_INTENT_KEYWORDS: list[str] = [
    "自动",
    "批处理",
    "全自动",
    "开始写",
    "开始创作",
    "启动",
    "run",
    "batch",
    "generate all",
    "开始生成",
    "完整流程",
    "流水线",
    "自动生成",
    "批量创作",
    "全流程",
]


def is_batch_mode_request(content: str) -> bool:
    """Detect if the user is requesting full batch pipeline mode.

    Checks the message content against batch intent keywords.
    This is checked BEFORE keyword/LLM routing so batch mode takes priority
    when the user wants to switch from conversation to automatic generation.

    Args:
        content: The user message content to analyze.

    Returns:
        True if the user is requesting batch pipeline mode.
    """
    content_lower = content.lower()
    for keyword in BATCH_INTENT_KEYWORDS:
        if keyword in content_lower:
            logger.debug("[ChatSupervisor] Batch mode keyword match: '%s'", keyword)
            return True
    return False


async def analyze_intent(state: dict[str, Any]) -> str:
    """Analyze user intent and determine the target sub-agent.

    Routing priority:
    1. Slash command (/write, /review, etc.)
    2. Keyword-based routing (fast path, no LLM call)
    3. LLM-based intent analysis (fallback for complex requests)

    Returns:
        Sub-agent name: "story_agent", "writing_agent", "review_agent", or "chat_agent"
    """
    messages = state.get("messages", [])
    if not messages:
        return "story_agent"  # First message, start with story planning

    last_message = messages[-1]
    content = _get_message_content(last_message)
    if not content:
        return "chat_agent"

    content = content.strip()

    # 1. Slash command routing
    if content.startswith("/"):
        cmd = content.split()[0].lower()
        agent = SLASH_COMMAND_MAP.get(cmd)
        if agent:
            logger.info("[ChatSupervisor] Slash command '%s' → %s", cmd, agent)
            return agent

    # 2. Batch mode detection (checked before keyword/LLM routing)
    # When the user requests full automatic generation, delegate to bridge_agent.
    if is_batch_mode_request(content):
        # Verify we have enough context to start batch mode
        has_setup = bool(state.get("world_setting") or state.get("story_outline"))
        if has_setup:
            logger.info("[ChatSupervisor] Batch mode request → bridge_agent")
            return "bridge_agent"
        else:
            logger.info("[ChatSupervisor] Batch mode requested but no setup context, routing to story_agent")
            return "story_agent"

    # 3. Keyword-based routing (fast path)
    agent = _keyword_route(content)
    if agent:
        logger.info("[ChatSupervisor] Keyword route → %s", agent)
        return agent

    # 4. LLM-based intent analysis (slow path)
    agent = await _llm_intent_analysis(content, state)
    logger.info("[ChatSupervisor] LLM intent analysis → %s", agent)
    return agent


def _get_message_content(msg: Any) -> str | None:
    """Extract text content from a message."""
    if hasattr(msg, "content"):
        return str(msg.content) if msg.content else None
    if isinstance(msg, dict):
        return msg.get("content")
    return str(msg) if msg else None


def _keyword_route(content: str) -> str | None:
    """Fast keyword-based routing without LLM call."""
    for agent, keywords in INTENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in content:
                return agent
    return None


async def _llm_intent_analysis(content: str, state: dict[str, Any]) -> str:
    """LLM-based intent analysis for complex or ambiguous requests."""
    try:
        llm = get_worker_llm()

        # Build context summary
        context = _build_context_summary(state)

        prompt = (
            "分析用户意图并路由到合适的 Agent。只输出 Agent 名称，不要其他内容。\n\n"
            "可选 Agent:\n"
            "- story_agent: 故事策划、世界观构建、角色设定、大纲规划\n"
            "- writing_agent: 章节写作、内容生成、续写\n"
            "- review_agent: 评审讨论、质量检查、修改建议\n"
            "- chat_agent: 普通对话、非创作类问题\n\n"
            f"用户消息: {content[:500]}\n"
            f"上下文: {context[:300]}\n\n"
            "Agent 名称:"
        )

        response = await llm.ainvoke(prompt)
        agent = _extract_agent_name(response)
        return agent
    except Exception as e:
        logger.warning("[ChatSupervisor] LLM intent analysis failed: %s", e)
        return "chat_agent"


def _build_context_summary(state: dict[str, Any]) -> str:
    """Build a brief summary of the current conversation context."""
    messages = state.get("messages", [])
    if not messages:
        return ""

    # Get last 2 human messages for context
    human_msgs = []
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            human_msgs.append(str(msg.content)[:100])
        if len(human_msgs) >= 2:
            break

    return " | ".join(reversed(human_msgs))


def _extract_agent_name(response: Any) -> str:
    """Extract the agent name from an LLM response."""
    text = response.content if hasattr(response, "content") else str(response)
    text = text.strip().lower()

    valid_agents = {"story_agent", "writing_agent", "review_agent", "chat_agent"}
    for agent in valid_agents:
        if agent in text:
            return agent
    return "chat_agent"