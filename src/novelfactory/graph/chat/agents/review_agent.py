"""Review sub-agent — provides quality assessment and suggestions via conversation."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from novelfactory.evaluation.service import get_review_service
from novelfactory.graph.chat.agents.utils import get_last_user_message
from novelfactory.state.lead_agent_state import LeadAgentState

logger = logging.getLogger(__name__)


async def review_agent_node(state: LeadAgentState) -> dict[str, Any]:
    """Review node — processes review requests from the user.

    Supports:
    - Reviewing the latest chapter draft
    - Discussing review results with the user
    - Providing specific improvement suggestions
    - Incorporating user feedback into revised recommendations
    """
    messages = state.get("messages", [])
    chapter_draft = state.get("chapter_draft", "")
    genre = state.get("genre", "")

    # Check if there's a chapter to review
    if not chapter_draft:
        # Check if review was requested via the last user message
        last_user_msg = get_last_user_message(messages)
        if last_user_msg and _is_review_request(last_user_msg):
            return {
                "current_agent": "review_agent",
                "messages": [
                    AIMessage(
                        content="我注意到你还没有章节内容可以评审。\n\n"
                        "请先让**写作助手**生成一章内容，或者直接粘贴你的章节文本过来，"
                        "我就可以帮你分析质量了！",
                        name="review_agent",
                    )
                ],
            }
        return {
            "current_agent": "review_agent",
            "messages": [
                AIMessage(
                    content="你好！我是评审编辑，可以帮你检查章节质量。\n\n"
                    "请提供你要评审的章节内容，或者先让写作助手生成一章。",
                    name="review_agent",
                )
            ],
        }

    last_user_msg = get_last_user_message(messages) or ""

    # If user has specific feedback, incorporate it
    user_feedback = ""
    if last_user_msg and not _is_review_request(last_user_msg):
        user_feedback = last_user_msg

    # Run the review service
    review_service = get_review_service()

    try:
        logger.info(
            "[ReviewAgent] Running review on chapter (%d chars)...",
            len(chapter_draft),
        )

        result = await review_service.evaluate_conversational(
            chapter_text=chapter_draft,
            genre=genre,
            user_feedback=user_feedback,
        )

        # Build a conversational review response
        response = _build_review_response(result, chapter_draft)

        return {
            "current_agent": "review_agent",
            "quality_score": result["final_score"],
            "messages": [AIMessage(content=response, name="review_agent")],
            "agent_context": {
                "type": "chapter_review",
                "score": result["final_score"],
            },
        }
    except Exception as e:
        logger.exception("[ReviewAgent] Review failed: %s", e)
        return {
            "current_agent": "review_agent",
            "messages": [
                AIMessage(
                    content=f"评审时遇到问题: {e}。请稍后重试。",
                    name="review_agent",
                )
            ],
        }


def _is_review_request(text: str) -> bool:
    """Check if the user message is requesting a review."""
    keywords = ["评审", "审核", "评价", "评分", "review", "检查", "质量", "看看"]
    return any(kw in text.lower() for kw in keywords)


def _build_review_response(result: dict, chapter_draft: str) -> str:
    """Build a conversational review response from the review result."""
    lines = [
        f"## 📊 评审结果\n",
        f"**{result['level_text']}** | 综合评分: **{result['final_score']:.1f}/100**\n",
        f"### 评分明细\n",
        f"- 四维质量: {result['quality_score']:.0f}/100",
        f"- AI 味指数: {result['ai_style_score']:.3f}（越低越好）",
        f"- 老书虫评分: {result['lao_shu_chong_score']:.0f}/100",
        f"- 跨章一致性: {result['cross_chapter_consistency']:.0f}/100",
    ]

    if result["debate_penalty"] > 0:
        lines.append(f"- 辩论惩罚: -{result['debate_penalty']:.1f}")

    # Summary
    lines.append(f"\n### 总结\n{result['summary']}\n")

    # Issues
    if result["toxic_points"]:
        lines.append("### ⚠️ 发现问题\n")
        for tp in result["toxic_points"]:
            lines.append(f"- {tp}")
        lines.append("")

    # Strengths
    if result["shuangdian_points"]:
        lines.append("### ✨ 亮点\n")
        for sp in result["shuangdian_points"]:
            lines.append(f"- {sp}")
        lines.append("")

    # Suggestions
    if result["suggestions"]:
        lines.append(f"### 💡 改进建议\n{result['suggestions']}\n")

    # Severe toxic warning
    if result["has_severe_toxic"]:
        lines.append("> ⚠️ **检测到严重问题**，建议仔细修改后重新评审。\n")

    # Interaction prompt
    lines.append(
        "---\n"
        "你可以：\n"
        "1. 要求我**详细分析**某个方面\n"
        "2. 告诉我想**如何修改**，我来出具体建议\n"
        "3. 修改后让我**重新评审**"
    )

    return "\n".join(lines)