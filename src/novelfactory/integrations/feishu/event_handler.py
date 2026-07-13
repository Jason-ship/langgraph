"""Feishu (Lark) event handler (synchronous).

Consumes stdout NDJSON from `lark-cli event consume` via pipeline:

    lark-cli event consume | python -m novelfactory.integrations.feishu.event_handler

All functions are synchronous and safe to use with interrupt recovery.

v6.5: Message sending routed through FeishuToolkit → httpx → tools-proxy,
delegating to feishu_api.send_lark_message().

v6.1 P1-7: 修复 asyncio.run 误用 — 使用模块级事件循环避免每次创建新循环。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid

import httpx

from novelfactory.config.constants import FEISHU_HTTPX_TIMEOUT, FEISHU_REQUESTS_TIMEOUT
from novelfactory.config.settings import settings
from novelfactory.crews.review_queue import review_queue
from novelfactory.integrations.feishu.feishu_api import send_lark_message

# v6.1: 模块级事件循环，避免反复 asyncio.run() 创建新循环
# 设计意图：循环与进程同生命周期，进程退出时由 OS 回收。
# _get_event_loop 已包含 is_closed() 恢复机制，无需显式关闭。
_event_loop: asyncio.AbstractEventLoop | None = None


def _get_event_loop() -> asyncio.AbstractEventLoop:
    """获取或创建模块级事件循环。"""
    global _event_loop
    if _event_loop is None or _event_loop.is_closed():
        _event_loop = asyncio.new_event_loop()
    return _event_loop


def _run_async(coro):
    """在模块级事件循环中运行协程。"""
    loop = _get_event_loop()
    return loop.run_until_complete(coro)


logger = logging.getLogger(__name__)

_FEISHU_HTTPX_TIMEOUT = (
    FEISHU_HTTPX_TIMEOUT  # httpx 客户端超时 — 唯一来源: config.constants
)
_FEISHU_REQUESTS_TIMEOUT = (
    FEISHU_REQUESTS_TIMEOUT  # 消息发送超时 — 唯一来源: config.constants
)
_ECHO_MAX_CHARS = 100  # 回显消息最大长度
_NEW_CMD_MAXSPLIT = (
    3  # /new 命令 split 最大分段数（得 4 段：cmd genre chapters name+seed）
)
_NEW_CMD_MIN_PARTS = 4  # /new 命令最小分段数

# chat_thread_mapping Redis key
_CHAT_THREAD_MAPPING_REDIS_KEY = "chat_thread_mapping"

# ── Mapping file path (fallback) ──────────────────────────────────────────────
_MAPPING_FILE_PATH = os.path.expanduser("~/.lark-cli/chat_thread_mapping.json")


# ── Intent Classification ─────────────────────────────────────────────────────

_FEISHU_APPROVE_KW = {
    "通过",
    "批准",
    "同意",
    "ok",
    "好",
    "可以",
    "yes",
    "approve",
    "继续",
}
_FEISHU_REJECT_KW = {"拒绝", "不行", "不好", "reject", "no", "停止", "不要"}
_FEISHU_MODIFY_KW = {"修改", "改一下", "改进", "rewrite", "modify", "modify:"}


def _classify_intent(text: str) -> str:
    """Classify a Feishu message into an intent.

    Order matters: check reject triggers (which may contain approve substrings
    like "不通过" contains "通过") BEFORE checking approve triggers.
    """
    t = text.lower().strip()
    if any(k in t for k in _FEISHU_REJECT_KW):
        return "reject"
    if any(k in t for k in _FEISHU_MODIFY_KW):
        return "modify"
    if any(k in t for k in _FEISHU_APPROVE_KW):
        return "approve"
    if t.startswith("/new "):
        return "new_project"
    return "chat"


# ── Chat ↔ Thread Mapping ─────────────────────────────────────────────────────


def _get_chat_thread_mapping() -> dict[str, str]:
    """Load the chat_id → thread_id mapping.

    Strategy:
      1. Try Redis first (async → module-level event loop)
      2. Fall back to file storage when Redis is unavailable
    """
    # Try Redis first
    try:
        from novelfactory.store.redis_store import get_redis_store

        store = get_redis_store()
        if store and store.available:
            data = _run_async(store.get_json(_CHAT_THREAD_MAPPING_REDIS_KEY))
            if isinstance(data, dict):
                return data
    except Exception as exc:
        logger.debug("[feishu] Redis unavailable for mapping read: %s", exc)

    # Fallback: file storage
    return _load_mapping_from_file()


def _save_chat_thread_mapping(mapping: dict[str, str]) -> None:
    """Persist the chat_id → thread_id mapping.

    Strategy:
      1. Save to Redis first (async → module-level event loop)
      2. Always save to file as backup
    """
    # Save to Redis
    try:
        from novelfactory.store.redis_store import get_redis_store

        store = get_redis_store()
        if store and store.available:
            _run_async(store.set_json(_CHAT_THREAD_MAPPING_REDIS_KEY, mapping))
    except Exception as exc:
        logger.debug("[feishu] Redis unavailable for mapping write: %s", exc)

    # Always write to file as backup
    _save_mapping_to_file(mapping)


def _load_mapping_from_file() -> dict[str, str]:
    """Load mapping from local file (fallback)."""
    if not os.path.exists(_MAPPING_FILE_PATH):
        return {}
    try:
        with open(_MAPPING_FILE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("[feishu] Failed to load mapping file: %s", exc)
        return {}


def _save_mapping_to_file(mapping: dict[str, str]) -> None:
    """Save mapping to local file (backup)."""
    try:
        os.makedirs(os.path.dirname(_MAPPING_FILE_PATH), exist_ok=True)
        with open(_MAPPING_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
    except OSError as exc:
        logger.warning("[feishu] Failed to save mapping file: %s", exc)


# ── Core Message Handlers ──────────────────────────────────────────────────────


def handle_feishu_message(event: dict) -> None:
    """Process a single Feishu message event (synchronous).

    This is the main entry point when running via stdin pipeline.
    Wrapped in try/except to prevent a single event crash from killing the pipe.
    """
    try:
        _handle_feishu_message_impl(event)
    except Exception:
        logger.error(
            "[feishu] Unhandled exception in handle_feishu_message", exc_info=True
        )


def _handle_feishu_message_impl(event: dict) -> None:
    """Core message handling logic."""
    message = event.get("event", {})
    try:
        content_obj = json.loads(message.get("content", "{}"))
    except (json.JSONDecodeError, TypeError):
        content_obj = {}

    text = content_obj.get("text", "").strip()
    chat_id = message.get("chat_id", "")
    sender_open_id = message.get("sender", {}).get("sender_id", {}).get("open_id", "")
    logger.debug("[feishu] sender_open_id=%s", sender_open_id)

    intent = _classify_intent(text)
    logger.info(f"[feishu] intent={intent} chat_id={chat_id[:20]}")

    if intent == "approve":
        _process_approve(chat_id, text)
    elif intent == "reject":
        _process_reject(chat_id, text)
    elif intent == "modify":
        _process_modify(chat_id, text)
    elif intent == "new_project":
        _process_new_project(chat_id, text)
    else:
        _forward_to_chatbot(chat_id, text)


def _process_approve(chat_id: str, text: str) -> None:
    """Handle an approval message."""
    thread_id = _get_chat_thread_mapping().get(chat_id)
    if not thread_id:
        _send_message(chat_id, "❌ 未找到关联项目")
        return

    review_queue.decide(thread_id, "approve", comment=text)
    _resume_workflow(thread_id, {"decision": "approve", "comment": text})
    _send_message(chat_id, f"✅ 已批准，Thread: {thread_id}")


def _process_reject(chat_id: str, text: str) -> None:
    """Handle a rejection message."""
    thread_id = _get_chat_thread_mapping().get(chat_id)
    if not thread_id:
        _send_message(chat_id, "❌ 未找到关联项目")
        return

    review_queue.decide(thread_id, "reject", comment=text)
    _resume_workflow(thread_id, {"decision": "reject", "comment": text})
    _send_message(chat_id, f"❌ 已拒绝，Thread: {thread_id}")


def _process_modify(chat_id: str, text: str) -> None:
    """Handle a modification request."""
    thread_id = _get_chat_thread_mapping().get(chat_id)
    if not thread_id:
        _send_message(chat_id, "❌ 未找到关联项目")
        return

    review_queue.decide(
        thread_id,
        "modify",
        comment=text,
        modifications={"feedback": text},
    )
    _resume_workflow(thread_id, {"decision": "modify", "comment": text})
    _send_message(chat_id, f"🔄 已记录修改意见，Thread: {thread_id}")


def _process_new_project(chat_id: str, text: str) -> None:
    """Handle /new project creation request.

    Expected format: /new <genre> <chapters> <name> <seed...>
    Example: /new 仙侠 10 仙逆 一个少年逆天改命的故事
    """
    parts = text.split(maxsplit=_NEW_CMD_MAXSPLIT)
    if len(parts) < _NEW_CMD_MIN_PARTS:
        _send_message(
            chat_id,
            "格式: /new <genre> <chapters> <name> <seed...>\n例: /new 仙侠 10 仙逆 一个少年逆天改命",
        )
        return

    _, genre, chapters_str, name_and_seed = parts
    name_parts = name_and_seed.split(maxsplit=1)
    name = name_parts[0]
    seed = name_parts[1] if len(name_parts) > 1 else ""

    try:
        chapters = int(chapters_str)
    except ValueError:
        _send_message(chat_id, f"❌ 章节数必须是数字: {chapters_str}")
        return

    thread_id = str(uuid.uuid4())

    # Save mapping
    mapping = _get_chat_thread_mapping()
    mapping[chat_id] = thread_id
    _save_chat_thread_mapping(mapping)

    # Call API to start workflow — 直接使用 settings.BASE_URL（默认 http://localhost:2024），
    # 不加 /api/v1 前缀（路由均为扁平路径如 /threads/{id}/runs）。
    base_url = settings.BASE_URL
    try:
        with httpx.Client(timeout=_FEISHU_HTTPX_TIMEOUT) as client:
            resp = client.post(
                f"{base_url}/threads/{thread_id}/runs",
                json={
                    "input": {
                        "seed_idea": seed,
                        "genre": genre,
                        "target_chapters": chapters,
                        "project_name": name,
                        "thread_id": thread_id,
                    },
                    "config": {"configurable": {"thread_id": thread_id}},
                },
            )
        if resp.status_code in (200, 201):
            _send_message(
                chat_id,
                f"✅ 项目已创建！\n"
                f"名称：{name}\n"
                f"类型：{genre}\n"
                f"目标：{chapters}章\n"
                f"Thread: {thread_id}",
            )
        else:
            _send_message(chat_id, f"❌ 创建失败 ({resp.status_code})")
    except Exception as e:
        _send_message(chat_id, f"❌ 创建失败: {str(e)[:100]}")


def _send_message(chat_id: str, text: str) -> None:
    """Send a Feishu text message via lark-cli subprocess.

    Delegates to feishu_api.send_lark_message() — the unified entry point.
    """
    id_type = "chat_id" if chat_id.startswith("oc_") else "open_id"
    send_lark_message(
        chat_id, text, receive_id_type=id_type, timeout=_FEISHU_REQUESTS_TIMEOUT
    )


def _forward_to_chatbot(chat_id: str, text: str) -> None:
    """Forward a general chat message for response."""
    # For now, echo back
    _send_message(chat_id, f"📝 收到: {text[:_ECHO_MAX_CHARS]}")


def _resume_workflow(thread_id: str, resume_data: dict) -> None:
    """Call the API to resume a paused workflow.

    不再调用不存在的 /api/v1/threads/{id}/resume 端点，而是复用标准的
    POST /threads/{thread_id}/runs 端点。streaming.py 的 _resolve_input_data
    会自动检测 interrupt，从 input.resume 提取数据并生成 Command(resume=...).
    """
    base_url = settings.BASE_URL
    try:
        with httpx.Client(timeout=_FEISHU_HTTPX_TIMEOUT) as client:
            resp = client.post(
                f"{base_url}/threads/{thread_id}/runs",
                json={
                    "input": {"resume": resume_data},
                    "config": {"configurable": {"thread_id": thread_id}},
                },
            )
            if resp.status_code not in (200, 201):
                logger.warning(
                    "[feishu] Resume returned HTTP %d for thread %s: %s",
                    resp.status_code,
                    thread_id,
                    resp.text[:200],
                )
    except Exception as e:
        logger.warning(f"[feishu] Failed to resume workflow {thread_id}: {e}")


# ── Stdin Pipeline Entry Point ────────────────────────────────────────────────


def main() -> None:
    """Read NDJSON from stdin and dispatch events."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("[feishu] Event handler started, reading stdin...")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            handle_feishu_message(event)
        except json.JSONDecodeError:
            logger.warning(f"[feishu] Skipping invalid JSON line: {line[:80]}")
        except Exception:
            logger.error(
                f"[feishu] Unexpected error processing event line: {line[:80]}",
                exc_info=True,
            )


if __name__ == "__main__":
    main()
