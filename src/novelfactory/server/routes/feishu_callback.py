"""飞书交互卡片回调 endpoint — 用于人工审核中断恢复。

v6.0: 接收飞书卡片按钮回调，恢复被 interrupt() 暂停的线程。

流程：
  1. wait_for_review_node 调用 interrupt() 暂停执行
  2. 中断数据通过 SSE 或飞书卡片展示给用户
  3. 用户点击飞书卡片按钮（approve/reject/modify）
  4. 飞书发送回调到此 endpoint
  5. endpoint 解析回调数据，调用 Command(resume={...}) 恢复线程

参考：LangGraph 官方 interrupt() + Command(resume=...) 模式
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from novelfactory.config.settings import settings
from novelfactory.server.deps import get_graph

logger = logging.getLogger(__name__)

# v6.1 P3-9: 持有后台任务引用，防止 GC 回收未完成的 asyncio.Task
_background_tasks: set[asyncio.Task] = set()

router = APIRouter(prefix="/feishu", tags=["feishu"])

_FEISHU_VERIFICATION_TOKEN = settings.FEISHU_VERIFICATION_TOKEN or os.environ.get(
    "FEISHU_VERIFICATION_TOKEN", ""
)
_FEISHU_ENCRYPT_KEY = settings.FEISHU_ENCRYPT_KEY or os.environ.get(
    "FEISHU_ENCRYPT_KEY", ""
)


async def _resume_thread(thread_id: str, resume_data: dict) -> bool:
    """恢复被 interrupt() 暂停的线程。

    Args:
        thread_id: 线程 ID
        resume_data: 传给 Command(resume=...) 的数据

    Returns:
        是否成功提交恢复请求
    """
    try:
        from novelfactory.config.constants import RECURSION_LIMIT
        from novelfactory.state.novel_context import NovelContext

        graph = await get_graph()
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": RECURSION_LIMIT,
        }
        context: NovelContext = {
            "thread_id": thread_id,
            "user_id": "",
            "project_id": "",
            "request_id": str(uuid.uuid4()),
            "lark_config": None,
        }

        # 非阻塞恢复：提交恢复请求到事件循环
        async def _do_resume() -> None:
            try:
                await graph.ainvoke(
                    Command(resume=resume_data), config=config, context=context
                )
                logger.info(
                    "[feishu_callback] Thread %s resumed with action=%s",
                    thread_id,
                    resume_data.get("action"),
                )
            except Exception as e:
                logger.error(
                    "[feishu_callback] Failed to resume thread %s: %s",
                    thread_id,
                    e,
                )

        # 在后台执行恢复（持有引用防止 GC 回收）
        task = asyncio.create_task(_do_resume())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        return True

    except Exception as e:
        logger.error("[feishu_callback] Resume setup failed: %s", e)
        return False


@router.post("/callback")
async def feishu_card_callback(request: Request) -> JSONResponse:
    """处理飞书交互卡片回调。

    飞书卡片按钮点击后会发送 POST 请求到此 endpoint。

    请求格式（飞书标准回调）：
    {
        "challenge": "...",  // URL 验证（首次配置时）
        "token": "...",
        "type": "url_verification" | "event_callback",
        "event": {
            "type": "card.action.trigger",
            "operator": {"open_id": "..."},
            "action": {
                "value": "{\"action\": \"approve\", \"thread_id\": \"...\"}",
                "tag": "button"
            }
        }
    }
    """
    body = await request.json()

    # URL 验证（飞书首次配置回调地址时）
    if body.get("type") == "url_verification":
        challenge = body.get("challenge", "")
        logger.info("[feishu_callback] URL verification: challenge=%s", challenge)
        return JSONResponse(content={"challenge": challenge})

    # Token 验证
    token = body.get("token", "")
    if _FEISHU_VERIFICATION_TOKEN and token != _FEISHU_VERIFICATION_TOKEN:
        logger.warning("[feishu_callback] Invalid token: %s", token[:8])
        return JSONResponse(status_code=403, content={"error": "invalid token"})

    # 处理卡片动作回调
    event = body.get("event", {})
    event_type = event.get("type", "")

    if event_type == "card.action.trigger":
        action = event.get("action", {})
        action_value = action.get("value", "")
        operator = event.get("operator", {})
        open_id = operator.get("open_id", "")

        logger.info(
            "[feishu_callback] Card action: value=%s, operator=%s",
            action_value[:100],
            open_id,
        )

        # 解析按钮 value
        from novelfactory.integrations.feishu.card_builder import (
            parse_card_action_value,
        )

        parsed = parse_card_action_value(action_value)
        if not parsed:
            return JSONResponse(
                status_code=400,
                content={"error": "invalid action value"},
            )

        thread_id = parsed.get("thread_id", "")
        action_type = parsed.get("action", "approve")

        if not thread_id:
            return JSONResponse(
                status_code=400,
                content={"error": "missing thread_id"},
            )

        # 构建 resume 数据
        resume_data: dict[str, Any] = {"action": action_type}

        # 如果是修改建议，预留 modifications 字段
        if action_type == "modify":
            # 飞书卡片回调不支持文本输入，先用默认修改建议
            # 后续可通过飞书消息回复获取具体修改内容
            resume_data["modifications"] = "用户通过飞书卡片请求修改"

        if action_type == "reject":
            resume_data["action"] = "reject"

        # 恢复线程
        success = await _resume_thread(thread_id, resume_data)

        if success:
            return JSONResponse(
                content={
                    "msg": "action processed",
                    "thread_id": thread_id,
                    "action": action_type,
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "failed to resume thread"},
            )

    # 未知事件类型
    logger.info("[feishu_callback] Unhandled event type: %s", event_type)
    return JSONResponse(content={"msg": "ok"})


@router.post("/resume/{thread_id}")
async def manual_resume(thread_id: str, request: Request) -> JSONResponse:
    """手动恢复线程（供调试和飞书机器人消息回调使用）。

    请求体：
    {
        "action": "approve" | "reject" | "modify" | "provide_guidance",
        "comment": "可选的用户评论",
        "modifications": "可选的修改建议"
    }
    """
    body = await request.json()
    action = body.get("action", "approve")

    resume_data: dict[str, Any] = {"action": action}
    if "comment" in body:
        resume_data["comment"] = body["comment"]
    if "modifications" in body:
        resume_data["modifications"] = body["modifications"]

    success = await _resume_thread(thread_id, resume_data)

    if success:
        return JSONResponse(
            content={
                "msg": "resume submitted",
                "thread_id": thread_id,
                "action": action,
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "failed to resume thread"},
        )
