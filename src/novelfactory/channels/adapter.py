"""Channel adapter — bridges legacy sync_crew/notify to the new channels layer.

Provides channel-aware notification helpers that prefer the FeishuChannel
when available, falling back to the existing FeishuToolkit → tools-proxy path.
"""

from __future__ import annotations

import logging
from typing import Any

from novelfactory.channels.message_bus import OutboundMessage
from novelfactory.channels.service import get_channel_service

logger = logging.getLogger(__name__)


async def send_channel_message(
    chat_id: str,
    text: str,
    *,
    thread_id: str = "",
    thread_ts: str | None = None,
    is_final: bool = True,
) -> bool:
    """Send a message through the Feishu channel layer when available.

    Falls back to False if the channel service is not running, so callers
    can use their existing fallback path (FeishuToolkit → tools-proxy).
    """
    service = get_channel_service()
    if service is None:
        logger.debug("[channel_adapter] Channel service not running, falling back")
        return False

    channel = service.get_channel("feishu")
    if channel is None or not channel.is_running:
        logger.debug("[channel_adapter] Feishu channel not running, falling back")
        return False

    msg = OutboundMessage(
        channel_name="feishu",
        chat_id=chat_id,
        thread_id=thread_id,
        text=text,
        is_final=is_final,
        thread_ts=thread_ts,
    )
    await service.bus.publish_outbound(msg)
    return True


async def send_chapter_complete_via_channel(
    chapter_num: int,
    quality_score: float,
    project_name: str,
    thread_id: str,
    feishu_doc_url: str = "",
    word_count: int = 0,
    chat_id: str | None = None,
) -> bool:
    """Send chapter completion notification through the Feishu channel.

    Returns True if sent via channel, False if caller should fall back.
    """
    if not chat_id:
        logger.debug("[channel_adapter] No chat_id for channel notification")
        return False

    score_emoji = "🟢" if quality_score >= 80 else "🟡" if quality_score >= 60 else "🔴"
    score_label = "优秀" if quality_score >= 80 else "良好" if quality_score >= 60 else "需改进"

    lines = [
        f"{score_emoji} 第{chapter_num}章已完成",
        "",
        f"【{project_name}】" if project_name else "",
        f"质量评分：{quality_score:.1f}/100 ({score_label})",
        f"字数：{word_count:,}" if word_count else "",
    ]
    if feishu_doc_url:
        lines.append(f"飞书文档：{feishu_doc_url}")

    text = "\n".join(line for line in lines if line)
    return await send_channel_message(chat_id, text, thread_id=thread_id)