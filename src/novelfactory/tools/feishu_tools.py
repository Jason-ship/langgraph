"""飞书工具集 — 消息/文档/通知/日历/通讯录/任务工具。

通过 @tool 装饰器封装 FeishuToolkit（→ httpx → tools-proxy HTTP API）的操作，
让 LLM Agent 能自主决定何时使用飞书各项能力。

使用方式：
    tools = get_feishu_tools()
    agent = create_react_agent(llm, tools=tools, prompt=...)

v6.5: 集成 FeishuToolkit（覆盖 21 域 209 方法），
    新增日历、通讯录、任务、妙记、知识库等工具绑定。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading

from langchain_core.tools import tool

from novelfactory.config.settings import settings
from novelfactory.integrations.feishu.feishu_toolkit import FeishuToolkit

logger = logging.getLogger(__name__)

# 延迟加载避免循环依赖
_FEISHU_TOOLKIT = None
_tk_lock = threading.Lock()


def _get_tk():
    global _FEISHU_TOOLKIT
    if _FEISHU_TOOLKIT is None:
        with _tk_lock:
            if _FEISHU_TOOLKIT is None:
                _FEISHU_TOOLKIT = FeishuToolkit()
    return _FEISHU_TOOLKIT


# ═════════════════════════════════════════════════════════════════════════════
# 消息发送（IM）
# ═════════════════════════════════════════════════════════════════════════════


@tool
def send_feishu_message(receive_id: str, text: str, id_type: str = "open_id") -> str:
    """发送飞书文本消息（个人或群聊）。

    v7.0: 优先使用渠道层（FeishuChannel WebSocket）发送，失败时降级到 HTTP API。

    通过 lark-cli 发送消息到飞书用户或群聊。
    适用于：通知用户章节完成、发送审核请求、推送进度报告。

    Args:
        receive_id: 接收者 ID（open_id 以 ou_ 开头，chat_id 以 oc_ 开头）
        text: 消息内容（纯文本）
        id_type: ID 类型，'open_id'（个人，默认）或 'chat_id'（群聊）
    """
    # Try channel layer first (only for chat_id type)
    if receive_id.startswith("oc_"):
        try:
            from novelfactory.channels.adapter import send_channel_message

            import asyncio
            loop = _get_tool_event_loop()
            sent = loop.run_until_complete(
                send_channel_message(receive_id, text)
            )
            if sent:
                return json.dumps({"status": "sent_via_channel", "to": receive_id}, ensure_ascii=False)
        except Exception:
            logger.debug("[feishu_tools] Channel layer unavailable, using HTTP fallback")

    # Fallback: FeishuToolkit → httpx → tools-proxy
    from novelfactory.integrations.feishu.feishu_api import send_lark_message

    try:
        if receive_id.startswith("oc_"):
            id_type = "chat_id"
        success = send_lark_message(receive_id, text, receive_id_type=id_type)
        if success:
            return json.dumps({"status": "sent", "to": receive_id}, ensure_ascii=False)
        return json.dumps({"status": "failed", "to": receive_id}, ensure_ascii=False)
    except Exception as e:
        logger.error("[feishu_tools] send_feishu_message error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# Module-level event loop for async bridge
_tool_event_loop: asyncio.AbstractEventLoop | None = None


def _get_tool_event_loop() -> asyncio.AbstractEventLoop:
    global _tool_event_loop
    if _tool_event_loop is None or _tool_event_loop.is_closed():
        _tool_event_loop = asyncio.new_event_loop()
    return _tool_event_loop


@tool
def send_feishu_card(receive_id: str, card_json: str, id_type: str = "open_id") -> str:
    """发送飞书交互卡片消息。

    发送包含按钮、进度条等交互元素的卡片消息。
    适用于：发送审核卡片（带通过/拒绝按钮）、发送进度卡片。

    Args:
        receive_id: 接收者 ID（open_id 以 ou_ 开头，chat_id 以 oc_ 开头）
        card_json: 卡片 JSON 字符串（飞书消息卡片格式）
        id_type: ID 类型，'open_id'（个人，默认）或 'chat_id'（群聊）
    """
    from novelfactory.integrations.feishu.feishu_api import send_lark_card

    try:
        if receive_id.startswith("oc_"):
            id_type = "chat_id"
        card = json.loads(card_json) if isinstance(card_json, str) else card_json
        success = send_lark_card(receive_id, card, receive_id_type=id_type)
        if success:
            return json.dumps(
                {"status": "card_sent", "to": receive_id}, ensure_ascii=False
            )
        return json.dumps({"status": "failed", "to": receive_id}, ensure_ascii=False)
    except Exception as e:
        logger.error("[feishu_tools] send_feishu_card error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def send_feishu_markdown(receive_id: str, markdown: str) -> str:
    """发送飞书 Markdown 格式消息。

    比纯文本支持更多排版（加粗、列表、链接）。
    适用于：发送带格式的通知、报告、汇总信息。

    Args:
        receive_id: 接收者 ID（open_id 以 ou_ 开头，chat_id 以 oc_ 开头）
        markdown: Markdown 格式消息内容
    """
    try:
        tk = _get_tk()
        if receive_id.startswith("oc_"):
            r = tk.im.send_markdown(markdown, chat_id=receive_id)
        else:
            r = tk.im.send_markdown(markdown, user_id=receive_id)
        if r.success:
            return json.dumps({"status": "sent", "to": receive_id}, ensure_ascii=False)
        return json.dumps({"status": "failed", "error": r.error}, ensure_ascii=False)
    except Exception as e:
        logger.error("[feishu_tools] send_feishu_markdown error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ═════════════════════════════════════════════════════════════════════════════
# 文档操作（Docs）
# ═════════════════════════════════════════════════════════════════════════════


@tool
def create_feishu_document(title: str, content: str, folder_token: str = "") -> str:
    """创建飞书在线文档。

    在指定目录下创建飞书文档，内容为 Markdown 格式。
    适用于：上传章节正文、创建设定文档、生成审核报告。

    Args:
        title: 文档标题
        content: 文档内容（Markdown 格式）
        folder_token: 目标文件夹 token（为空则使用 FEISHU_ROOT_FOLDER 环境变量）
    """
    from novelfactory.integrations.feishu.feishu_api import create_feishu_doc

    if not folder_token:
        # v6.1: 统一从 settings 读取
        folder_token = settings.FEISHU_ROOT_FOLDER or os.environ.get(
            "FEISHU_ROOT_FOLDER", ""
        )
    if not folder_token:
        return json.dumps(
            {"error": "未指定 folder_token 且 FEISHU_ROOT_FOLDER 未配置"},
            ensure_ascii=False,
        )
    try:
        url = create_feishu_doc(title, content, folder_token)
        if url:
            return json.dumps(
                {"status": "created", "url": url, "title": title}, ensure_ascii=False
            )
        return json.dumps({"status": "failed", "title": title}, ensure_ascii=False)
    except Exception as e:
        logger.error("[feishu_tools] create_feishu_document error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def fetch_feishu_document(doc_token: str) -> str:
    """读取飞书在线文档内容。

    获取指定飞书文档的 Markdown 格式内容。
    适用于：读取已上传的章节文档、审核设定文档、查看同步状态。

    Args:
        doc_token: 文档 token（doxcn 开头）
    """
    try:
        tk = _get_tk()
        r = tk.docs.fetch(doc_token)
        if r.success:
            return json.dumps(r.data, ensure_ascii=False, default=str)
        return json.dumps({"error": r.error}, ensure_ascii=False)
    except Exception as e:
        logger.error("[feishu_tools] fetch_feishu_document error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def search_feishu_docs(query: str) -> str:
    """搜索飞书文档。

    在全站范围内搜索标题或内容匹配的飞书文档。
    适用于：查找项目相关文档、定位已上传的章节。

    Args:
        query: 搜索关键词（文档标题或内容）
    """
    try:
        tk = _get_tk()
        r = tk.docs.search(query)
        if r.success:
            return json.dumps(r.data, ensure_ascii=False, default=str)
        return json.dumps({"error": r.error}, ensure_ascii=False)
    except Exception as e:
        logger.error("[feishu_tools] search_feishu_docs error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ═════════════════════════════════════════════════════════════════════════════
# 文件夹与云盘
# ═════════════════════════════════════════════════════════════════════════════


@tool
def ensure_feishu_project_folders(project_name: str) -> str:
    """幂等创建飞书项目标准目录树。

    创建：{FEISHU_ROOT_FOLDER}/{project_name}/设定文档/ 和 正文/卷N/ 目录结构。
    返回各文件夹的 token，后续上传文档时需要使用。

    Args:
        project_name: 项目名称
    """
    from novelfactory.integrations.feishu.feishu_api import (
        ensure_project_folders_idempotent,
    )

    try:
        tokens = ensure_project_folders_idempotent(project_name)
        return json.dumps(tokens, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error("[feishu_tools] ensure_feishu_project_folders error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def upload_chapter_to_feishu(
    project_name: str,
    chapter_number: int,
    chapter_text: str,
    volume_number: int,
    folder_tokens_json: str,
) -> str:
    """将章节内容上传为飞书在线文档。

    自动定位到正确的卷文件夹，创建带标题的飞书文档。
    适用于：章节完成后自动同步到飞书。

    Args:
        project_name: 项目名称
        chapter_number: 章节号
        chapter_text: 章节正文内容
        volume_number: 卷号
        folder_tokens_json: 文件夹 token JSON 字符串（由 ensure_feishu_project_folders 返回）
    """
    from novelfactory.integrations.feishu.feishu_api import upload_chapter_as_doc

    try:
        folder_tokens = (
            json.loads(folder_tokens_json)
            if isinstance(folder_tokens_json, str)
            else folder_tokens_json
        )
        url = upload_chapter_as_doc(
            project_name, chapter_number, chapter_text, volume_number, folder_tokens
        )
        if url:
            return json.dumps(
                {"status": "uploaded", "url": url, "chapter": chapter_number},
                ensure_ascii=False,
            )
        return json.dumps(
            {"status": "failed", "chapter": chapter_number}, ensure_ascii=False
        )
    except Exception as e:
        logger.error("[feishu_tools] upload_chapter_to_feishu error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def search_feishu_files(query: str) -> str:
    """搜索飞书云盘文件。

    在全站范围内搜索云盘中的文件。
    适用于：查找已上传的设定文档、素材文件。

    Args:
        query: 搜索关键词
    """
    try:
        tk = _get_tk()
        r = tk.drive.search(query)
        if r.success:
            return json.dumps(r.data, ensure_ascii=False, default=str)
        return json.dumps({"error": r.error}, ensure_ascii=False)
    except Exception as e:
        logger.error("[feishu_tools] search_feishu_files error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ═════════════════════════════════════════════════════════════════════════════
# 通讯录
# ═════════════════════════════════════════════════════════════════════════════


@tool
def search_feishu_user(query: str) -> str:
    """搜索飞书通讯录用户。

    按姓名、邮箱、手机号搜索企业通讯录中的用户。
    适用于：查找审核人、确认用户 open_id、获取联系人信息。

    Args:
        query: 搜索关键词（姓名/邮箱/手机号）
    """
    try:
        tk = _get_tk()
        r = tk.contact.search_user(query)
        if r.success:
            return json.dumps(r.data, ensure_ascii=False, default=str)
        return json.dumps({"error": r.error}, ensure_ascii=False)
    except Exception as e:
        logger.error("[feishu_tools] search_feishu_user error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ═════════════════════════════════════════════════════════════════════════════
# 任务管理
# ═════════════════════════════════════════════════════════════════════════════


@tool
def create_feishu_task(
    summary: str, assignee: str = "", due: str = "", description: str = ""
) -> str:
    """在飞书创建任务。

    创建一个新的飞书任务，可指定负责人和截止时间。
    适用于：创建创作计划任务、指派章节审核任务、设置截止提醒。

    Args:
        summary: 任务标题（必填）
        assignee: 负责人 open_id（可选，ou_xxx 格式）
        due: 截止时间（可选，支持格式：+3d, YYYY-MM-DD, ISO 8601）
        description: 任务描述（可选）
    """
    try:
        tk = _get_tk()
        r = tk.task.create(summary, assignee=assignee, due=due, description=description)
        if r.success:
            return json.dumps(r.data, ensure_ascii=False, default=str)
        return json.dumps({"error": r.error}, ensure_ascii=False)
    except Exception as e:
        logger.error("[feishu_tools] create_feishu_task error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def get_my_feishu_tasks() -> str:
    """获取我的飞书任务列表。

    返回当前登录用户的待办任务清单。
    适用于：查看待办事项、确认进度安排。
    """
    try:
        tk = _get_tk()
        r = tk.task.get_my_tasks()
        if r.success:
            return json.dumps(r.data, ensure_ascii=False, default=str)
        return json.dumps({"error": r.error}, ensure_ascii=False)
    except Exception as e:
        logger.error("[feishu_tools] get_my_feishu_tasks error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ═════════════════════════════════════════════════════════════════════════════
# 日历
# ═════════════════════════════════════════════════════════════════════════════


@tool
def get_feishu_agenda(date: str = "") -> str:
    """查看飞书日历日程。

    获取指定日期的飞书日历日程列表。
    适用于：查看当日安排、确认空闲时间、规划创作时间线。

    Args:
        date: 日期（可选，YYYY-MM-DD 格式，为空则今天）
    """
    try:
        tk = _get_tk()
        r = tk.calendar.agenda(date=date)
        if r.success:
            return json.dumps(r.data, ensure_ascii=False, default=str)
        return json.dumps({"error": r.error}, ensure_ascii=False)
    except Exception as e:
        logger.error("[feishu_tools] get_feishu_agenda error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ═════════════════════════════════════════════════════════════════════════════
# 妙记
# ═════════════════════════════════════════════════════════════════════════════


@tool
def get_feishu_minutes_summary(minute_token: str) -> str:
    """获取飞书妙记总结。

    从飞书妙记（会议录制）中提取 AI 总结。
    适用于：获取会议纪要、提取讨论要点、跟踪创作讨论。

    Args:
        minute_token: 妙记 token
    """
    try:
        tk = _get_tk()
        r = tk.minutes.summary(minute_token)
        if r.success:
            return json.dumps(r.data, ensure_ascii=False, default=str)
        return json.dumps({"error": r.error}, ensure_ascii=False)
    except Exception as e:
        logger.error("[feishu_tools] get_feishu_minutes_summary error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ═════════════════════════════════════════════════════════════════════════════
# 审批
# ═════════════════════════════════════════════════════════════════════════════


@tool
def list_feishu_approval_tasks() -> str:
    """查询飞书审批任务列表。

    获取待处理的审批任务。
    适用于：查看待审批的审核请求、确认审批进度。
    """
    try:
        tk = _get_tk()
        r = tk.approval.list_tasks()
        if r.success:
            return json.dumps(r.data, ensure_ascii=False, default=str)
        return json.dumps({"error": r.error}, ensure_ascii=False)
    except Exception as e:
        logger.error("[feishu_tools] list_feishu_approval_tasks error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ═════════════════════════════════════════════════════════════════════════════
# 审核通知
# ═════════════════════════════════════════════════════════════════════════════


@tool
def send_review_request(
    thread_id: str,
    review_type: str,
    project_name: str,
    content_summary: str,
    doc_url: str = "",
) -> str:
    """发送飞书审核请求（阻塞等待人工审核）。

    通过飞书通知触发人工审核流程。审核结果通过 interrupt/resume 机制返回。
    适用于：章节质量审核、大纲审核、设定审核。

    Args:
        thread_id: 对话线程 ID
        review_type: 审核类型（'chapter' / 'outline' / 'setup'）
        project_name: 项目名称
        content_summary: 内容摘要（显示在审核通知中）
        doc_url: 审核文档 URL（可选）
    """
    from novelfactory.integrations.feishu.notify import send_review_notification

    try:
        send_review_notification(
            thread_id=thread_id,
            review_type=review_type,
            project_name=project_name,
            content_summary=content_summary,
            doc_url=doc_url or None,
        )
        return json.dumps(
            {"status": "review_sent", "thread_id": thread_id}, ensure_ascii=False
        )
    except Exception as e:
        logger.error("[feishu_tools] send_review_request error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ═════════════════════════════════════════════════════════════════════════════
# 工具集导出
# ═════════════════════════════════════════════════════════════════════════════


def get_feishu_tools() -> list:
    """返回飞书工具列表，可直接传入 create_react_agent。"""
    return [
        # 消息（IM）
        send_feishu_message,
        send_feishu_markdown,
        send_feishu_card,
        # 文档（Docs）
        create_feishu_document,
        fetch_feishu_document,
        search_feishu_docs,
        # 云盘（Drive）
        ensure_feishu_project_folders,
        upload_chapter_to_feishu,
        search_feishu_files,
        # 通讯录（Contact）
        search_feishu_user,
        # 任务（Task）
        create_feishu_task,
        get_my_feishu_tasks,
        # 日历（Calendar）
        get_feishu_agenda,
        # 妙记（Minutes）
        get_feishu_minutes_summary,
        # 审批（Approval）
        list_feishu_approval_tasks,
        # 审核通知
        send_review_request,
    ]


def get_feishu_tools_basic() -> list:
    """返回基础飞书工具（轻量级，减少 token 消耗）。

    适用于不需要完整工具箱的简单场景。
    """
    return [
        send_feishu_message,
        send_feishu_card,
        create_feishu_document,
        ensure_feishu_project_folders,
        upload_chapter_to_feishu,
        send_review_request,
    ]
