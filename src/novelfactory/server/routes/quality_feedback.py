"""飞书质量反馈 Webhook 端点。

接收飞书质量反馈，调用 QualityTunerAgent 处理，回复调参结果。

支持两种消息格式：
1. 飞书事件订阅格式（event_callback / card.action.trigger）
2. 简单 JSON 格式（自定义集成）

简单格式请求体示例：
{
    "feedback": "最近写的章节AI味太重了",
    "thread_id": "项目线程ID（可选，用于时间旅行验证）",
    "chat_id": "飞书群聊 ID（可选，用于回复）"
}

版本：v1.0.0
创建日期：2026-08-12
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage

from novelfactory.config.settings import settings
from novelfactory.state.lead_agent_state import LeadAgentState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feishu", tags=["feishu"])

_FEISHU_VERIFICATION_TOKEN = settings.FEISHU_VERIFICATION_TOKEN or os.environ.get(
    "FEISHU_VERIFICATION_TOKEN", ""
)

# 持有后台任务引用，防止 GC 回收未完成的 asyncio.Task
_background_tasks: set[asyncio.Task] = set()


# ═══════════════════════════════════════════════════════════════════════════════
#  反馈处理
# ═══════════════════════════════════════════════════════════════════════════════


async def _process_feedback(feedback: str, thread_id: str = "") -> dict[str, Any]:
    """处理反馈，调用 QualityTunerAgent。

    Returns:
        {"changes": [...], "explanation": str, "replay_summary": str}
    """
    from novelfactory.graph.chat.agents.quality_tuner_agent import (
        quality_tuner_agent_node,
    )

    state: LeadAgentState = {
        "messages": [HumanMessage(content=feedback)],
        "thread_id": thread_id or f"quality-tuner-{uuid.uuid4().hex[:8]}",
        "current_agent": "quality_tuner_agent",
    }

    result = await quality_tuner_agent_node(state)

    # 提取回复文本
    explanation = ""
    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        explanation = getattr(last, "content", "") if hasattr(last, "content") else str(last)

    # 提取参数变更
    from novelfactory.config.quality_params import quality_center
    history = quality_center.get_history(limit=5)
    changes = [h for h in history if h.get("operator") == "quality_tuner"]

    return {
        "changes": changes,
        "explanation": explanation,
        "replay_summary": "",
    }


async def _send_feishu_reply(chat_id: str, text: str) -> None:
    """通过飞书回复消息（复用 channels.adapter，降级到 FeishuToolkit）。"""
    try:
        from novelfactory.channels.adapter import send_channel_message

        ok = await send_channel_message(chat_id=chat_id, text=text)
        if ok:
            return
    except Exception as e:
        logger.warning("[QualityFeedback] Channel reply failed: %s", e)

    try:
        from novelfactory.integrations.feishu.feishu_api import feishu_api

        feishu_api.send_lark_message(chat_id=chat_id, text=text)
    except Exception as e:
        logger.warning("[QualityFeedback] FeishuToolkit reply failed: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
#  请求解析
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_feedback(body: dict[str, Any]) -> str:
    """从不同消息格式中提取反馈文本。"""
    # 简单格式
    feedback = body.get("feedback", "")
    if feedback:
        return str(feedback)

    # 飞书事件订阅格式
    event = body.get("event", {})
    if isinstance(event, dict):
        # 消息事件
        msg = event.get("message", {})
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if isinstance(content, str):
                try:
                    import json
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        return str(parsed.get("text", ""))
                except json.JSONDecodeError:
                    return content
        # 卡片动作
        action = event.get("action", {})
        if isinstance(action, dict):
            value = action.get("value", "")
            if isinstance(value, str):
                try:
                    import json
                    parsed = json.loads(value)
                    if isinstance(parsed, dict):
                        return str(parsed.get("feedback", "") or parsed.get("text", ""))
                except json.JSONDecodeError:
                    return value

    # 通用兜底
    for key in ("text", "content", "message"):
        val = body.get(key, "")
        if val:
            return str(val)
    return ""


def _extract_chat_id(body: dict[str, Any]) -> str:
    """从请求体中提取飞书 chat_id。"""
    chat_id = body.get("chat_id", "")
    if chat_id:
        return str(chat_id)

    event = body.get("event", {})
    if isinstance(event, dict):
        msg = event.get("message", {})
        if isinstance(msg, dict):
            chat_id = msg.get("chat_id", "")
            if chat_id:
                return str(chat_id)
        chat_id = event.get("chat_id", "")
        if chat_id:
            return str(chat_id)
    return ""


def _extract_thread_id(body: dict[str, Any]) -> str:
    """从请求体中提取 LangGraph thread_id。"""
    thread_id = body.get("thread_id", "")
    if thread_id:
        return str(thread_id)

    event = body.get("event", {})
    if isinstance(event, dict):
        action = event.get("action", {})
        if isinstance(action, dict):
            value = action.get("value", "")
            if isinstance(value, str):
                try:
                    import json
                    parsed = json.loads(value)
                    if isinstance(parsed, dict):
                        thread_id = parsed.get("thread_id", "")
                        if thread_id:
                            return str(thread_id)
                except json.JSONDecodeError:
                    pass
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
#  端点
# ═══════════════════════════════════════════════════════════════════════════════


@router.post("/quality-feedback")
async def quality_feedback_webhook(request: Request) -> JSONResponse:
    """接收飞书质量反馈 Webhook，调用 QualityTunerAgent 处理。

    流程：
      1. 提取反馈文本
      2. 调用 quality_tuner_agent_node（自动定位参数 + 变更 + 可选时间旅行验证）
      3. 通过飞书回复结果
    """
    try:
        body = await request.json()
    except Exception as e:
        logger.warning("[QualityFeedback] Invalid JSON body: %s", e)
        return JSONResponse(status_code=400, content={"error": "invalid JSON"})

    # URL 验证（飞书首次配置回调地址时）
    if body.get("type") == "url_verification":
        challenge = body.get("challenge", "")
        logger.info("[QualityFeedback] URL verification: challenge=%s", challenge)
        return JSONResponse(content={"challenge": challenge})

    # Token 验证
    token = body.get("token", "")
    if _FEISHU_VERIFICATION_TOKEN and token and token != _FEISHU_VERIFICATION_TOKEN:
        logger.warning("[QualityFeedback] Invalid token: %s", token[:8])
        return JSONResponse(status_code=403, content={"error": "invalid token"})

    # 提取反馈文本
    feedback_text = _extract_feedback(body)
    if not feedback_text:
        logger.info("[QualityFeedback] No feedback text in body: %s", str(body)[:200])
        return JSONResponse(status_code=400, content={"error": "no feedback text"})

    thread_id = _extract_thread_id(body)
    chat_id = _extract_chat_id(body)

    logger.info(
        "[QualityFeedback] Received feedback: %s (thread_id=%s)",
        feedback_text[:100], thread_id or "N/A",
    )

    # 异步处理（防止长评审阻塞 Webhook 响应）
    async def _handle() -> None:
        try:
            result = await _process_feedback(feedback_text, thread_id)

            # 通过飞书回复
            if chat_id and result.get("explanation"):
                await _send_feishu_reply(chat_id, result["explanation"])
        except Exception as e:
            logger.exception("[QualityFeedback] Processing failed: %s", e)
            if chat_id:
                await _send_feishu_reply(
                    chat_id, f"⚠️ 质量反馈处理失败: {e}"
                )

    task = asyncio.create_task(_handle())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return JSONResponse(
        content={"msg": "processing started", "thread_id": thread_id or ""}
    )


@router.post("/quality-params")
async def quality_params_direct(request: Request) -> JSONResponse:
    """直接查看/调整质量参数（程序化接口，不经过 LLM Agent）。

    请求体示例（查看）：
    {}

    请求体示例（调整）：
    {"updates": {"verdict.pass_threshold": 80.0}, "reason": "用户要求提高通过线"}

    请求体示例（回滚）：
    {"rollback_change_id": "abc123"}
    """
    try:
        body = await request.json()
    except Exception as e:
        logger.warning("[QualityFeedback] Invalid JSON body: %s", e)
        return JSONResponse(status_code=400, content={"error": "invalid JSON"})

    from novelfactory.config.quality_params import quality_center

    # 查看快照
    if not body or "updates" not in body and "rollback_change_id" not in body:
        return JSONResponse(content={
            "params": quality_center.get_all(),
            "history": quality_center.get_history(limit=20),
        })

    # 回滚
    if "rollback_change_id" in body:
        ok = quality_center.rollback(str(body["rollback_change_id"]))
        return JSONResponse(content={
            "rollback": ok,
            "history": quality_center.get_history(limit=20),
        })

    # 调整
    updates = body.get("updates", {})
    reason = body.get("reason", "direct API")
    if not isinstance(updates, dict) or not updates:
        return JSONResponse(status_code=400, content={"error": "no updates"})

    results = quality_center.update_many(updates, reason=reason, operator="api")
    return JSONResponse(content={
        "applied": {k: quality_center.get(k) for k, v in results.items() if v},
        "rejected": [k for k, v in results.items() if not v],
        "history": quality_center.get_history(limit=20),
    })
