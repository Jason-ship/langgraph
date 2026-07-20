"""Feishu notification sender — prefers new channels layer, falls back to FeishuToolkit.

v6.5: All message sending routed through FeishuToolkit (httpx HTTP proxy).
v7.0: Channels layer preferred when available (FeishuChannel WebSocket).
"""

from __future__ import annotations

import asyncio
import logging
import os

from novelfactory.config.settings import settings as _st
from novelfactory.crews.review_queue import ReviewItem, ReviewStatus, review_queue
from novelfactory.integrations.feishu.feishu_api import send_lark_message

logger = logging.getLogger(__name__)

# 常量
_FEISHU_TIMEOUT = 10  # 飞书消息发送超时（秒）
_NOTIFY_SUMMARY_LEN = 200  # 通知内容摘要最大长度
_NOTIFY_QUEUE_LEN = 500  # 入队内容摘要最大长度
_QUALITY_EXCELLENT = 80  # 优秀质量阈值
_QUALITY_GOOD = 60  # 良好质量阈值
_PCT_BASE = 100  # 百分比基数
_HTTP_OK = 200  # HTTP 200 OK

# Module-level event loop for async → sync bridge
_event_loop: asyncio.AbstractEventLoop | None = None


def _get_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create a module-level event loop for async bridge calls."""
    global _event_loop
    if _event_loop is None or _event_loop.is_closed():
        _event_loop = asyncio.new_event_loop()
    return _event_loop


def _send_lark_message(receive_id: str, text: str) -> bool:
    """Send a text message via FeishuToolkit → httpx → tools-proxy.

    自动检测 receive_id 类型：以 'oc_' 开头视为 chat_id，否则视为 open_id。
    """
    id_type = "chat_id" if receive_id.startswith("oc_") else "open_id"
    return send_lark_message(
        receive_id, text, receive_id_type=id_type, timeout=_FEISHU_TIMEOUT
    )


def send_review_notification(
    thread_id: str,
    review_type: str,
    project_name: str,
    content_summary: str,
    doc_url: str | None = None,
    chat_id: str | None = None,
    full_content: str | None = None,
) -> None:
    """Send a review request to Feishu and register it in the queue.

    If full_content is longer than _NOTIFY_SUMMARY_LEN (200), it will be
    uploaded as a Feishu document first, and the notification will contain
    the document URL instead of truncated text.
    """
    # 如果内容需要审核且长度超过摘要限制，创建飞书文档
    final_doc_url = doc_url
    if full_content and len(full_content) > _NOTIFY_SUMMARY_LEN:
        try:
            from novelfactory.integrations.feishu.feishu_api import create_feishu_doc

            # v6.1: 统一从 settings 读取
            root_token = _st.FEISHU_ROOT_FOLDER or os.environ.get(
                "FEISHU_ROOT_FOLDER", ""
            )
            if root_token:
                doc_title = f"{project_name} - {review_type}审核内容"
                doc_md = f"# {doc_title}\n\n{full_content}"
                created_url = create_feishu_doc(doc_title, doc_md, root_token)
                if created_url:
                    final_doc_url = created_url
                    logger.info(
                        "[notify] Created review doc %s for %s",
                        created_url,
                        review_type,
                    )
        except Exception as e:
            logger.warning("[notify] Failed to create review doc: %s", e)

    lines = [
        f"📋 待审核：{review_type}",
        "",
        f"项目：{project_name}",
    ]
    # 有文档链接时显示完整链接 + 简要描述
    if final_doc_url:
        lines.append(f"内容：{content_summary[:_NOTIFY_SUMMARY_LEN]}")
        lines.append(f"📄 完整内容：{final_doc_url}")
    else:
        lines.append(f"内容：{content_summary[:_NOTIFY_SUMMARY_LEN]}")
    if doc_url:
        lines.append(f"📄 文档链接：{doc_url}")

    msg_text = "\n".join(lines)

    # v6.1: 统一从 settings 读取
    target_id = (
        chat_id or _st.FEISHU_USER_OPEN_ID or os.environ.get("FEISHU_USER_OPEN_ID", "")
    )
    if not target_id:
        logger.warning(
            "[notify] No target for review notification — review NOT enqueued"
        )
        return

    success = _send_lark_message(target_id, msg_text)
    if not success:
        logger.error(
            "[notify] Failed to send review notification — review NOT enqueued"
        )
        return

    review_queue.add(
        ReviewItem(
            thread_id=thread_id,
            review_type=review_type,
            project_name=project_name,
            current_chapter=1,
            content_summary=content_summary[:_NOTIFY_QUEUE_LEN],
            feishu_doc_url=doc_url,
            status=ReviewStatus.PENDING,
        )
    )
    logger.info(
        f"[notify] Review notification sent and enqueued: {review_type} thread={thread_id}"
    )


def send_progress_notification(
    thread_id: str,
    chapter: int,
    total: int,
    chat_id: str | None = None,
) -> None:
    """Send a progress update notification.

    v7.0: Prefers channels layer when available.
    """
    progress = chapter / max(total, 1) * _PCT_BASE
    message = (
        f"📖 创作进度：第{chapter}章完成 ({progress:.0f}%)\n"
        f"目标：{total}章\n"
        f"Thread: {thread_id}"
    )
    # Try channel layer first
    if chat_id:
        try:
            from novelfactory.channels.adapter import send_channel_message

            loop = _get_event_loop()
            sent = loop.run_until_complete(
                send_channel_message(chat_id, message, thread_id=thread_id)
            )
            if sent:
                return
        except Exception:
            logger.debug("[notify] Channel layer unavailable for progress", exc_info=True)

    # Fallback
    target = (
        chat_id or _st.FEISHU_USER_OPEN_ID or os.environ.get("FEISHU_USER_OPEN_ID", "")
    )
    if target:
        _send_lark_message(target, message)


def send_human_intervention_alert(
    chapter_num: int,
    quality_score: float,
    project_name: str,
    thread_id: str,
    custom_message: str = "",
    chat_id: str | None = None,
) -> None:
    """Send a high-priority alert when a chapter needs human guidance.

    v7.0: Prefers channels layer when available.
    """
    header = f"🚨 【需人工介入】第{chapter_num}章质量评分 {quality_score:.1f}/100（自动重写已用尽）"
    body = custom_message or f"【{project_name}】请提供具体的修改指导后回复。"
    lines = [
        header,
        "",
        body,
        f"Thread: {thread_id}" if thread_id else "",
    ]
    message_text = "\n".join(line for line in lines if line)

    # Try channel layer first
    if chat_id:
        try:
            from novelfactory.channels.adapter import send_channel_message

            loop = _get_event_loop()
            sent = loop.run_until_complete(
                send_channel_message(chat_id, message_text, thread_id=thread_id)
            )
            if sent:
                return
        except Exception:
            logger.debug("[notify] Channel layer unavailable for alert", exc_info=True)

    # Fallback
    target = (
        chat_id or _st.FEISHU_USER_OPEN_ID or os.environ.get("FEISHU_USER_OPEN_ID", "")
    )
    if target:
        _send_lark_message(target, message_text)
    else:
        logger.warning("[notify] No FEISHU_USER_OPEN_ID configured, skipping send")


def _send_chapter_complete_fallback(
    chapter_num: int,
    quality_score: float,
    usage: dict,
    project_name: str = "",
    thread_id: str = "",
    feishu_doc_url: str = "",
    word_count: int = 0,
) -> None:
    """Fallback: send chapter completion via FeishuToolkit → tools-proxy."""
    score_emoji = (
        "🟢"
        if quality_score >= _QUALITY_EXCELLENT
        else "🟡"
        if quality_score >= _QUALITY_GOOD
        else "🔴"
    )
    score_label = (
        "优秀"
        if quality_score >= _QUALITY_EXCELLENT
        else "良好"
        if quality_score >= _QUALITY_GOOD
        else "需改进"
    )

    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

    lines = [
        f"{score_emoji} 第{chapter_num}章已完成",
        "",
        f"【{project_name}】" if project_name else "",
        f"质量评分：{quality_score:.1f}/100 ({score_label})",
        f"字数：{word_count:,}" if word_count else "",
        f"Token 消耗：输入 {input_tokens:,} / 输出 {output_tokens:,} / 合计 {total_tokens:,}",
    ]
    if feishu_doc_url:
        lines.append(f"飞书文档：{feishu_doc_url}")
    lines.append(f"Thread: {thread_id}" if thread_id else "")

    message_text = "\n".join(line for line in lines if line)

    target = (
        _st.FEISHU_CHAT_ID
        or os.environ.get("FEISHU_CHAT_ID", "")
        or _st.FEISHU_USER_OPEN_ID
        or os.environ.get("FEISHU_USER_OPEN_ID", "")
    )
    if target:
        id_type = "chat_id" if target.startswith("oc_") else "open_id"
        send_lark_message(target, text=message_text, receive_id_type=id_type, timeout=_FEISHU_TIMEOUT)
    else:
        logger.warning("[notify] No FEISHU_USER_OPEN_ID or FEISHU_CHAT_ID configured, skipping send")


def send_chapter_complete_notification(
    chapter_num: int,
    quality_score: float,
    usage: dict,
    project_name: str = "",
    thread_id: str = "",
    chat_id: str | None = None,
    feishu_doc_url: str = "",
    word_count: int = 0,
) -> None:
    """Send a chapter completion notification with quality score, token usage, and doc link.

    v7.0: Prefers channels layer (FeishuChannel WebSocket) when available,
    falls back to FeishuToolkit → tools-proxy.
    """
    # Try channel layer first (async → sync bridge via event loop)
    if chat_id or _st.FEISHU_CHAT_ID or os.environ.get("FEISHU_CHAT_ID", ""):
        target_chat_id = chat_id or _st.FEISHU_CHAT_ID or os.environ.get("FEISHU_CHAT_ID", "")
        try:
            from novelfactory.channels.adapter import send_chapter_complete_via_channel

            loop = _get_event_loop()
            sent = loop.run_until_complete(
                send_chapter_complete_via_channel(
                    chapter_num=chapter_num,
                    quality_score=quality_score,
                    project_name=project_name,
                    thread_id=thread_id,
                    feishu_doc_url=feishu_doc_url,
                    word_count=word_count,
                    chat_id=target_chat_id,
                )
            )
            if sent:
                logger.info("[notify] Chapter %d notification sent via channel layer", chapter_num)
                return
        except Exception:
            logger.debug("[notify] Channel layer unavailable, falling back to FeishuToolkit", exc_info=True)

    # Fallback: FeishuToolkit → tools-proxy
    _send_chapter_complete_fallback(
        chapter_num=chapter_num,
        quality_score=quality_score,
        usage=usage,
        project_name=project_name,
        thread_id=thread_id,
        feishu_doc_url=feishu_doc_url,
        word_count=word_count,
    )
