"""Review / Human-in-the-loop nodes: wait_for_review + chapter_human_guidance.

v6.0: 集成飞书交互卡片
  - wait_for_review_node 在 interrupt() 前发送飞书交互卡片
  - 用户可在飞书中直接点击 approve/reject/modify 按钮
  - 回调通过 /feishu/callback endpoint 恢复线程
"""

from __future__ import annotations

import logging
import os

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from novelfactory.config.llm import get_worker_llm
from novelfactory.integrations.feishu.card_builder import build_review_card
from novelfactory.integrations.feishu.feishu_api import send_lark_card
from novelfactory.integrations.feishu.notify import send_progress_notification
from novelfactory.state.novel_state import NovelFactoryState

logger = logging.getLogger(__name__)

_DRAFT_PREVIEW_LENGTH = 500


def _send_review_card(
    thread_id: str,
    interrupt_data: dict,
    chat_id: str | None = None,
) -> None:
    """发送飞书交互审核卡片。

    优先发送到 chat_id（群聊），其次发送到 thread_id 对应的用户。
    """
    card = build_review_card(interrupt_data)

    # 优先级: chat_id（群聊） > FEISHU_USER_OPEN_ID（环境变量）
    # v6.5-fix: 不再使用 thread_id 作为 fallback — UUID 不是合法的飞书 ID
    # v6.1: 统一从 settings 读取
    from novelfactory.config.settings import settings as _st

    target_id = (
        chat_id or _st.FEISHU_USER_OPEN_ID or os.environ.get("FEISHU_USER_OPEN_ID", "")
    )
    if not target_id:
        logger.warning(
            "[review] 缺少 chat_id 且 FEISHU_USER_OPEN_ID 未配置，跳过卡片发送"
        )
        return

    id_type = "chat_id" if target_id.startswith("oc_") else "open_id"
    try:
        send_lark_card(
            receive_id=target_id,
            card=card,
            receive_id_type=id_type,
        )
        logger.info("[review] Feishu review card sent to %s", target_id)
    except Exception as e:
        logger.warning("[review] Feishu review card failed: %s", e)


def wait_for_review_node(state: NovelFactoryState) -> dict:
    """Suspend execution via standard interrupt() until a review decision arrives.

    v6.0: 在 interrupt() 前发送飞书交互卡片，用户可直接在飞书中审核。

    The interrupt payload is surfaced to the caller via stream.interrupts
    (when using astream_events with version="v2"). The caller resumes
    execution by passing Command(resume={...}) with the same thread_id.
    """
    # ── Feishu: notify user that review is needed ────────────────────────────────
    current_ch = state.get("current_chapter", 1)
    target_ch = state.get("target_chapters", 1)
    project_name = state.get(
        "project_name",
        state.get("project_context", {}).get("project_name", "未命名项目"),
    )
    thread_id = state.get("thread_id", "")
    chat_id = state.get("chat_id", "")
    try:
        send_progress_notification(
            thread_id=thread_id,
            chapter=current_ch,
            total=target_ch,
            chat_id=chat_id or None,
        )
    except Exception as e:
        logger.warning("[wait_for_review] Feishu notification failed: %s", e)

    review_type = state.get("pending_review", "unknown")
    chapter_text = state.get("chapter_draft", "") or state.get("refined_chapter", "")
    draft_preview = chapter_text[:_DRAFT_PREVIEW_LENGTH] if chapter_text else ""
    review_result = state.get("review_result", {})
    quality_score = review_result.get("quality_score", 0) or state.get(
        "quality_score", 0
    )

    interrupt_data = {
        "review_type": review_type,
        "project_name": project_name,
        "chapter_id": current_ch,
        "chapter_draft_preview": draft_preview,
        "quality_score": float(quality_score),
        "suggested_actions": ["approve", "request_changes", "regenerate"],
        "thread_id": thread_id,
    }

    # ── 发送飞书交互卡片 ──────────────────────────────────────────────────────
    _send_review_card(thread_id, interrupt_data, chat_id)

    # ── Standard interrupt — suspends execution until Command(resume=...) ─────
    decision = interrupt(interrupt_data)

    # When resumed via Command(resume={...}), decision is the resume value
    if isinstance(decision, dict):
        action = decision.get("action", "approve")
        if action == "approve":
            return {
                "user_decision": "approve",
                "pending_review": None,
                "chapter_approved": True,
            }
        if action == "reject":
            return {
                "user_decision": "reject",
                "pending_review": None,
                "chapter_approved": False,
            }
        if action == "modify":
            return {
                "user_decision": "modify",
                "pending_review": None,
                "modifications": decision.get("modifications"),
            }
        if action == "provide_guidance":
            return {
                "user_decision": "provide_guidance",
                "pending_review": None,
                "human_guidance": decision.get("comment", ""),
                "chapter_needs_guidance": False,
            }

    # Default: approve
    return {
        "user_decision": "approve",
        "pending_review": None,
        "chapter_approved": True,
    }


def chapter_human_guidance(state: NovelFactoryState) -> dict:
    """全自动 LLM 指导：低分章节耗尽重试后自动生成修改建议并递交。"""
    cr = state.get("crew_result", {})
    current_ch = state.get("current_chapter", 1)
    review_result = cr.get("review_result", {})
    quality_score = review_result.get("quality_score", 0)
    review_comments = review_result.get("review_comments", "")
    chapter_text = cr.get("refined_chapter", "") or cr.get("chapter_draft", "")

    # 使用 LLM 自动生成修改指导
    try:
        llm = get_worker_llm()
        guidance_prompt = (
            f"第{current_ch}章质量评分 {quality_score:.1f}/100，所有自动重写次数已用尽。\n\n"
            f"审核意见：{review_comments[:500]}\n\n"
            f"章节正文（前800字）：\n{chapter_text[:800]}\n\n"
            f"请分析章节问题并提供具体的修改指导（50-200字）。"
        )
        response = llm.invoke([HumanMessage(content=guidance_prompt)])
        guidance_text = (
            response.content if hasattr(response, "content") else str(response)
        )
    except Exception as e:
        logger.warning("[chapter_human_guidance] LLM 生成指导失败: %s", e)
        guidance_text = f"第{current_ch}章质量评分 {quality_score:.1f}/100。请检查审核意见并优化章节。{review_comments[:200]}"

    logger.info(
        "[chapter_human_guidance] chapter=%d, score=%.1f → auto guidance generated",
        current_ch,
        quality_score,
    )

    return {
        "user_decision": "provide_guidance",
        "pending_review": None,
        "human_guidance": guidance_text,
        "chapter_needs_guidance": False,
        "guidance_complete": True,
    }
