"""Feishu/Lark channel — connects to Feishu via WebSocket (no public IP needed).

Migrated from DeerFlow app/channels/feishu.py.
Adapted for NovelFactory: uses existing FeishuToolkit for card/API calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Literal

from novelfactory.channels.base import Channel
from novelfactory.channels.commands import is_known_channel_command, strip_leading_mentions
from novelfactory.channels.connection_identity import attach_connection_identity
from novelfactory.channels.message_bus import (
    PENDING_CLARIFICATION_METADATA_KEY,
    RESOLVED_FROM_PENDING_CLARIFICATION_METADATA_KEY,
    InboundMessage,
    InboundMessageType,
    MessageBus,
    OutboundMessage,
    ResolvedAttachment,
)

logger = logging.getLogger(__name__)
PENDING_CLARIFICATION_TTL_SECONDS = 30 * 60
FEISHU_INBOUND_BATCH_WINDOW_SECONDS = 0.75
SOURCE_PREVIEW_METADATA_KEY = "feishu_source_preview"


def _is_feishu_command(text: str) -> bool:
    return is_known_channel_command(text)


class FeishuChannel(Channel):
    """Feishu/Lark IM channel using the ``lark-oapi`` WebSocket client.

    Configuration keys:
        - ``app_id``: Feishu app ID.
        - ``app_secret``: Feishu app secret.

    Message flow:
        1. User sends a message → bot adds "OK" emoji reaction
        2. Bot replies with a card: "Working on it......"
        3. Agent processes the message and returns a result
        4. Bot updates the card with the result
        5. Bot adds "DONE" emoji reaction to the original message
    """

    def __init__(self, bus: MessageBus, config: dict[str, Any]) -> None:
        super().__init__(name="feishu", bus=bus, config=config)
        self._thread: threading.Thread | None = None
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._api_client = None
        self._lark = None
        self._CreateMessageReactionRequest = None
        self._CreateMessageReactionRequestBody = None
        self._Emoji = None
        self._PatchMessageRequest = None
        self._PatchMessageRequestBody = None
        self._background_tasks: set[asyncio.Task] = set()
        self._running_card_ids: dict[str, str] = {}
        self._running_card_tasks: dict[str, asyncio.Task] = {}
        self._pending_clarifications: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._pending_inbound_batches: dict[tuple[str, str], dict[str, Any]] = {}
        self._CreateFileRequest = None
        self._CreateFileRequestBody = None
        self._CreateImageRequest = None
        self._CreateImageRequestBody = None
        self._GetMessageResourceRequest = None
        self._thread_lock = threading.Lock()

    @staticmethod
    def _non_empty_str(value: Any) -> str | None:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    @staticmethod
    def _pending_key(chat_id: str, user_id: str) -> tuple[str, str]:
        return (chat_id, user_id)

    @staticmethod
    def _should_include_source_preview(*, chat_type: str | None, root_id: str | None, parent_id: str | None, thread_id: str | None) -> bool:
        if chat_type == "p2p":
            return False
        return bool(root_id or parent_id or thread_id)

    @staticmethod
    def _compact_source_preview(text: str) -> str | None:
        stripped = text.strip()
        if not stripped:
            return None
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if not lines:
            return None
        preview = "\n".join(lines[:3])
        if len(preview) > 240:
            preview = preview[:237].rstrip() + "..."
        return preview

    @classmethod
    def _compose_card_text(cls, text: str, metadata: dict[str, Any] | None = None) -> str:
        preview = None
        if isinstance(metadata, dict):
            raw_preview = metadata.get(SOURCE_PREVIEW_METADATA_KEY)
            if isinstance(raw_preview, str) and raw_preview.strip():
                preview = raw_preview.strip()
        if not preview:
            return text
        quoted_preview = "\n".join(f"> {line}" for line in preview.splitlines())
        return f"{quoted_preview}\n\n{text}"

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def is_running(self) -> bool:
        if not self._running:
            return False
        return self._thread is not None and self._thread.is_alive()

    def _build_event_handler(self, lark):
        return (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(self._on_message)
            .register_p2_im_message_message_read_v1(self._on_ignored_message_event)
            .register_p2_im_message_reaction_created_v1(self._on_ignored_message_event)
            .register_p2_im_message_reaction_deleted_v1(self._on_ignored_message_event)
            .register_p2_im_message_recalled_v1(self._on_ignored_message_event)
            .build()
        )

    async def start(self) -> None:
        if self._running:
            return

        try:
            import lark_oapi as lark
            from lark_oapi.api.im.v1 import (
                CreateFileRequest,
                CreateFileRequestBody,
                CreateImageRequest,
                CreateImageRequestBody,
                CreateMessageReactionRequest,
                CreateMessageReactionRequestBody,
                CreateMessageRequest,
                CreateMessageRequestBody,
                Emoji,
                GetMessageResourceRequest,
                PatchMessageRequest,
                PatchMessageRequestBody,
                ReplyMessageRequest,
                ReplyMessageRequestBody,
            )
        except ImportError:
            logger.error("lark-oapi is not installed. Install it with: pip install lark-oapi")
            return

        self._lark = lark
        self._CreateMessageRequest = CreateMessageRequest
        self._CreateMessageRequestBody = CreateMessageRequestBody
        self._ReplyMessageRequest = ReplyMessageRequest
        self._ReplyMessageRequestBody = ReplyMessageRequestBody
        self._CreateMessageReactionRequest = CreateMessageReactionRequest
        self._CreateMessageReactionRequestBody = CreateMessageReactionRequestBody
        self._Emoji = Emoji
        self._PatchMessageRequest = PatchMessageRequest
        self._PatchMessageRequestBody = PatchMessageRequestBody
        self._CreateFileRequest = CreateFileRequest
        self._CreateFileRequestBody = CreateFileRequestBody
        self._CreateImageRequest = CreateImageRequest
        self._CreateImageRequestBody = CreateImageRequestBody
        self._GetMessageResourceRequest = GetMessageResourceRequest

        app_id = self.config.get("app_id", "")
        app_secret = self.config.get("app_secret", "")
        domain = self.config.get("domain", "https://open.feishu.cn")

        if not app_id or not app_secret:
            logger.error("Feishu channel requires app_id and app_secret")
            return

        self._api_client = lark.Client.builder().app_id(app_id).app_secret(app_secret).domain(domain).build()
        logger.info("[Feishu] using domain: %s", domain)
        self._main_loop = asyncio.get_event_loop()

        self._running = True
        self.bus.subscribe_outbound(self._on_outbound)

        self._thread = threading.Thread(
            target=self._run_ws,
            args=(app_id, app_secret, domain),
            daemon=True,
        )
        self._thread.start()
        logger.info("Feishu channel started")

    def _run_ws(self, app_id: str, app_secret: str, domain: str) -> None:
        """Construct and run the lark WS client in a thread with a fresh event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            import lark_oapi as lark
            import lark_oapi.ws.client as _ws_client_mod

            _ws_client_mod.loop = loop

            event_handler = self._build_event_handler(lark)
            ws_client = lark.ws.Client(
                app_id=app_id,
                app_secret=app_secret,
                event_handler=event_handler,
                log_level=lark.LogLevel.INFO,
                domain=domain,
            )
            ws_client.start()
        except Exception:
            if self._running:
                logger.exception("Feishu WebSocket error")
            self._running = False

    def _on_ignored_message_event(self, event) -> None:
        logger.debug("[Feishu] ignoring non-content message event: %s", type(event).__name__)

    async def stop(self) -> None:
        self._running = False
        self.bus.unsubscribe_outbound(self._on_outbound)
        for task in list(self._background_tasks):
            task.cancel()
        self._background_tasks.clear()
        for task in list(self._running_card_tasks.values()):
            task.cancel()
        self._running_card_tasks.clear()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Feishu channel stopped")

    # -- send / outbound ---------------------------------------------------

    async def send(self, msg: OutboundMessage, *, _max_retries: int = 3) -> None:
        if not self._api_client:
            logger.warning("[Feishu] send called but no api_client available")
            return

        logger.info("[Feishu] sending reply: chat_id=%s, thread_ts=%s, text_len=%d", msg.chat_id, msg.thread_ts, len(msg.text))
        await self._send_with_retry(
            lambda: self._send_card_message(msg),
            max_retries=_max_retries,
            log_prefix="[Feishu]",
        )

    async def send_file(self, msg: OutboundMessage, attachment: ResolvedAttachment) -> bool:
        if not self._api_client:
            return False

        if attachment.is_image and attachment.size > 10 * 1024 * 1024:
            logger.warning("[Feishu] image too large (%d bytes), skipping: %s", attachment.size, attachment.filename)
            return False
        if not attachment.is_image and attachment.size > 30 * 1024 * 1024:
            logger.warning("[Feishu] file too large (%d bytes), skipping: %s", attachment.size, attachment.filename)
            return False

        try:
            if attachment.is_image:
                file_key = await self._upload_image(attachment.actual_path)
                msg_type = "image"
                content = json.dumps({"image_key": file_key})
            else:
                file_key = await self._upload_file(attachment.actual_path, attachment.filename)
                msg_type = "file"
                content = json.dumps({"file_key": file_key})

            if msg.thread_ts:
                request = self._ReplyMessageRequest.builder().message_id(msg.thread_ts).request_body(
                    self._ReplyMessageRequestBody.builder().msg_type(msg_type).content(content).build()
                ).build()
                await asyncio.to_thread(self._api_client.im.v1.message.reply, request)
            else:
                request = self._CreateMessageRequest.builder().receive_id_type("chat_id").request_body(
                    self._CreateMessageRequestBody.builder().receive_id(msg.chat_id).msg_type(msg_type).content(content).build()
                ).build()
                await asyncio.to_thread(self._api_client.im.v1.message.create, request)

            logger.info("[Feishu] file sent: %s (type=%s)", attachment.filename, msg_type)
            return True
        except Exception:
            logger.exception("[Feishu] failed to upload/send file: %s", attachment.filename)
            return False

    async def _upload_image(self, path: Path) -> str:
        with open(str(path), "rb") as f:
            request = self._CreateImageRequest.builder().request_body(
                self._CreateImageRequestBody.builder().image_type("message").image(f).build()
            ).build()
            response = await asyncio.to_thread(self._api_client.im.v1.image.create, request)
        if not response.success():
            raise RuntimeError(f"Feishu image upload failed: code={response.code}, msg={response.msg}")
        return response.data.image_key

    async def _upload_file(self, path: Path, filename: str) -> str:
        suffix = path.suffix.lower() if hasattr(path, "suffix") else ""
        if suffix in (".xls", ".xlsx", ".csv"):
            file_type = "xls"
        elif suffix in (".ppt", ".pptx"):
            file_type = "ppt"
        elif suffix == ".pdf":
            file_type = "pdf"
        elif suffix in (".doc", ".docx"):
            file_type = "doc"
        else:
            file_type = "stream"

        with open(str(path), "rb") as f:
            request = self._CreateFileRequest.builder().request_body(
                self._CreateFileRequestBody.builder().file_type(file_type).file_name(filename).file(f).build()
            ).build()
            response = await asyncio.to_thread(self._api_client.im.v1.file.create, request)
        if not response.success():
            raise RuntimeError(f"Feishu file upload failed: code={response.code}, msg={response.msg}")
        return response.data.file_key

    async def receive_file(self, msg: InboundMessage, thread_id: str, *, user_id: str | None = None) -> InboundMessage:
        if not msg.thread_ts:
            logger.warning("[Feishu] received file message without thread_ts")
            return msg
        files = msg.files
        if not files:
            return msg
        text = msg.text
        for file in files:
            if file.get("image_key"):
                virtual_path = await self._receive_single_file(msg.thread_ts, file["image_key"], "image", thread_id, user_id=user_id)
                text = text.replace("[image]", virtual_path, 1)
            elif file.get("file_key"):
                virtual_path = await self._receive_single_file(msg.thread_ts, file["file_key"], "file", thread_id, user_id=user_id)
                text = text.replace("[file]", virtual_path, 1)
        msg.text = text
        return msg

    async def _receive_single_file(self, message_id: str, file_key: str, type: Literal["image", "file"], thread_id: str, *, user_id: str | None = None) -> str:
        request = self._GetMessageResourceRequest.builder().message_id(message_id).file_key(file_key).type(type).build()

        def inner():
            return self._api_client.im.v1.message_resource.get(request)

        try:
            response = await asyncio.to_thread(inner)
        except Exception:
            logger.exception("[Feishu] resource get request failed for key=%s type=%s", file_key, type)
            return f"Failed to obtain the [{type}]"

        if not response.success():
            logger.warning("[Feishu] resource get failed: key=%s, type=%s, code=%s", file_key, type, response.code)
            return f"Failed to obtain the [{type}]"

        image_stream = getattr(response, "file", None)
        if image_stream is None:
            logger.warning("[Feishu] resource get returned no file stream")
            return f"Failed to obtain the [{type}]"

        try:
            content: bytes = await asyncio.to_thread(image_stream.read)
        except Exception:
            logger.exception("[Feishu] failed to read resource stream")
            return f"Failed to obtain the [{type}]"

        if not content:
            logger.warning("[Feishu] empty resource content")
            return f"Failed to obtain the [{type}]"

        # Save to local uploads directory
        # TODO: 使用 settings.storage_path_expanded 替代相对路径，避免工作目录依赖
        uploads_dir = Path(".novelfactory") / "uploads" / thread_id
        uploads_dir.mkdir(parents=True, exist_ok=True)

        ext = "png" if type == "image" else "bin"
        raw_filename = getattr(response, "file_name", "") or f"feishu_{file_key[-12:]}.{ext}"

        if "." in raw_filename:
            name_part, ext = raw_filename.rsplit(".", 1)
            name_part = re.sub(r"[./\\]", "_", name_part)
            filename = f"{name_part}.{ext}"
        else:
            filename = re.sub(r"[./\\]", "_", raw_filename)
        resolved_target = uploads_dir / filename

        def down_load():
            with self._thread_lock:
                resolved_target.write_bytes(content)

        try:
            await asyncio.to_thread(down_load)
        except Exception:
            logger.exception("[Feishu] failed to persist downloaded resource: %s", resolved_target)
            return f"Failed to obtain the [{type}]"

        virtual_path = f"/mnt/user-data/uploads/{resolved_target.name}"
        logger.info("[Feishu] downloaded resource: key=%s -> %s", file_key, virtual_path)
        return virtual_path

    # -- card formatting ---------------------------------------------------

    @staticmethod
    def _build_card_content(text: str) -> str:
        """Build a Feishu interactive card with markdown content."""
        card = {
            "config": {"wide_screen_mode": True, "update_multi": True},
            "elements": [{"tag": "markdown", "content": text}],
        }
        return json.dumps(card)

    # -- reaction helpers --------------------------------------------------

    async def _add_reaction(self, message_id: str, emoji_type: str = "THUMBSUP") -> None:
        if not self._api_client or not self._CreateMessageReactionRequest:
            return
        try:
            request = self._CreateMessageReactionRequest.builder().message_id(message_id).request_body(
                self._CreateMessageReactionRequestBody.builder().reaction_type(
                    self._Emoji.builder().emoji_type(emoji_type).build()
                ).build()
            ).build()
            await asyncio.to_thread(self._api_client.im.v1.message_reaction.create, request)
            logger.info("[Feishu] reaction '%s' added to message %s", emoji_type, message_id)
        except Exception:
            logger.exception("[Feishu] failed to add reaction '%s'", emoji_type)

    async def _reply_card(self, message_id: str, text: str) -> str | None:
        if not self._api_client:
            return None
        content = self._build_card_content(text)
        request = self._ReplyMessageRequest.builder().message_id(message_id).request_body(
            self._ReplyMessageRequestBody.builder().msg_type("interactive").content(content).build()
        ).build()
        response = await asyncio.to_thread(self._api_client.im.v1.message.reply, request)
        response_data = getattr(response, "data", None)
        return getattr(response_data, "message_id", None)

    async def _create_card(self, chat_id: str, text: str) -> None:
        if not self._api_client:
            return
        content = self._build_card_content(text)
        request = self._CreateMessageRequest.builder().receive_id_type("chat_id").request_body(
            self._CreateMessageRequestBody.builder().receive_id(chat_id).msg_type("interactive").content(content).build()
        ).build()
        await asyncio.to_thread(self._api_client.im.v1.message.create, request)

    async def _update_card(self, message_id: str, text: str) -> None:
        if not self._api_client or not self._PatchMessageRequest:
            return
        content = self._build_card_content(text)
        request = self._PatchMessageRequest.builder().message_id(message_id).request_body(
            self._PatchMessageRequestBody.builder().content(content).build()
        ).build()
        await asyncio.to_thread(self._api_client.im.v1.message.patch, request)

    # -- running card management -------------------------------------------

    async def _create_running_card(self, source_message_id: str, text: str, *, metadata: dict[str, Any] | None = None) -> str | None:
        running_card_id = await self._reply_card(source_message_id, self._compose_card_text(text, metadata))
        if running_card_id:
            self._running_card_ids[source_message_id] = running_card_id
            logger.info("[Feishu] running card created: source=%s card=%s", source_message_id, running_card_id)
        return running_card_id

    def _ensure_running_card_started(self, source_message_id: str, text: str = "thinking...", *, metadata: dict[str, Any] | None = None) -> asyncio.Task | None:
        if source_message_id in self._running_card_ids:
            return None
        if source_message_id in self._running_card_tasks:
            return self._running_card_tasks[source_message_id]
        task = asyncio.create_task(self._create_running_card(source_message_id, text, metadata=metadata))
        self._running_card_tasks[source_message_id] = task
        return task

    async def _ensure_running_card(self, source_message_id: str, text: str = "thinking...", *, metadata: dict[str, Any] | None = None) -> str | None:
        running_card_id = self._running_card_ids.get(source_message_id)
        if running_card_id:
            return running_card_id
        task = self._ensure_running_card_started(source_message_id, text, metadata=metadata)
        if task is None:
            return self._running_card_ids.get(source_message_id)
        return await task

    async def _send_card_message(self, msg: OutboundMessage) -> None:
        source_message_id = msg.thread_ts
        if source_message_id:
            running_card_id = self._running_card_ids.get(source_message_id)
            if not running_card_id:
                running_card_task = self._running_card_tasks.get(source_message_id)
                if running_card_task:
                    running_card_id = await running_card_task

            if running_card_id:
                card_text = self._compose_card_text(msg.text, msg.metadata)
                try:
                    await self._update_card(running_card_id, card_text)
                except Exception:
                    if not msg.is_final:
                        raise
                    logger.exception("[Feishu] failed to patch running card, falling back to final reply")
                    await self._reply_card(source_message_id, card_text)
                logger.info("[Feishu] running card updated: source=%s card=%s", source_message_id, running_card_id)
            elif msg.is_final:
                await self._reply_card(source_message_id, self._compose_card_text(msg.text, msg.metadata))
            else:
                await self._ensure_running_card(source_message_id, msg.text, metadata=msg.metadata)

            if msg.is_final:
                self._running_card_ids.pop(source_message_id, None)
                await self._add_reaction(source_message_id, "DONE")
            return

        await self._create_card(msg.chat_id, msg.text)

    # -- thread mapping ----------------------------------------------------

    def _remember_thread_mapping(self, msg: OutboundMessage, *topic_ids: str | None) -> None:
        store = self.config.get("channel_store")
        if store is None or not msg.thread_id:
            return
        metadata_topic_ids = [
            msg.metadata.get("message_id"),
            msg.metadata.get("root_id"),
            msg.metadata.get("parent_id"),
            msg.metadata.get("thread_id"),
            msg.metadata.get("topic_id"),
        ]
        user_id = ""
        raw_user_id = msg.metadata.get("user_id")
        if isinstance(raw_user_id, str):
            user_id = raw_user_id

        seen: set[str] = set()
        for topic_id in [*topic_ids, *metadata_topic_ids]:
            topic_id = self._non_empty_str(topic_id)
            if not topic_id or topic_id in seen:
                continue
            seen.add(topic_id)
            try:
                store.set_thread_id(self.name, msg.chat_id, msg.thread_id, topic_id=topic_id, user_id=user_id)
            except Exception:
                logger.exception("[Feishu] failed to remember thread mapping for topic_id=%s", topic_id)

    def _remember_pending_clarification(self, msg: OutboundMessage, card_message_id: str | None) -> None:
        if not msg.is_final or msg.metadata.get(PENDING_CLARIFICATION_METADATA_KEY) is not True:
            return
        user_id = self._non_empty_str(msg.metadata.get("user_id"))
        topic_id = self._non_empty_str(msg.metadata.get("topic_id"))
        source_message_id = self._non_empty_str(msg.thread_ts) or self._non_empty_str(msg.metadata.get("message_id"))
        if not (user_id and topic_id and msg.thread_id and source_message_id and card_message_id):
            return
        key = self._pending_key(msg.chat_id, user_id)
        pending = {
            "thread_id": msg.thread_id,
            "topic_id": topic_id,
            "source_message_id": source_message_id,
            "card_message_id": card_message_id,
            "created_at": time.time(),
        }
        with self._thread_lock:
            self._pending_clarifications.setdefault(key, []).append(pending)

    def _consume_pending_clarification(self, chat_id: str, user_id: str) -> dict[str, Any] | None:
        key = self._pending_key(chat_id, user_id)
        with self._thread_lock:
            pending_items = self._pending_clarifications.get(key)
            if not pending_items:
                return None
            now = time.time()
            while pending_items:
                pending = pending_items.pop(0)
                created_at = pending.get("created_at")
                if isinstance(created_at, (int, float)) and now - created_at <= PENDING_CLARIFICATION_TTL_SECONDS:
                    if pending_items:
                        self._pending_clarifications[key] = pending_items
                    else:
                        self._pending_clarifications.pop(key, None)
                    return pending
            self._pending_clarifications.pop(key, None)
            return None

    def _resolve_topic_id(self, chat_id: str, msg_id: str, *, root_id: str | None, parent_id: str | None, thread_id: str | None) -> tuple[str, bool]:
        store = self.config.get("channel_store")
        candidates = [root_id, parent_id, thread_id]

        if store is not None:
            for candidate in candidates:
                candidate = self._non_empty_str(candidate)
                if not candidate:
                    continue
                try:
                    if store.get_thread_id(self.name, chat_id, topic_id=candidate):
                        return candidate, True
                except Exception:
                    logger.exception("[Feishu] failed to resolve stored topic mapping")
        return root_id or msg_id, False

    @staticmethod
    def _is_batchable_file_inbound(*, msg_type: InboundMessageType, text: str, files: list[dict[str, Any]], root_id: str | None, parent_id: str | None, thread_id: str | None) -> bool:
        return msg_type == InboundMessageType.CHAT and text in {"[file]", "[image]"} and len(files) == 1 and not (root_id or parent_id or thread_id)

    # -- inbound message processing ----------------------------------------

    def _schedule_prepare_inbound(self, msg_id: str, inbound: InboundMessage, *, source_message_ids: list[str] | None = None) -> None:
        if self._main_loop and self._main_loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(
                self._prepare_inbound(msg_id, inbound, source_message_ids=source_message_ids),
                self._main_loop,
            )
            fut.add_done_callback(lambda f, mid=msg_id: self._log_future_error(f, "prepare_inbound", mid))

    def _schedule_batch_flush(self, key: tuple[str, str], source_message_id: str) -> None:
        if self._main_loop and self._main_loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self._flush_pending_inbound_batch_after(key, source_message_id), self._main_loop)
            fut.add_done_callback(lambda f, mid=source_message_id: self._log_future_error(f, "flush_inbound_batch", mid))

    def _queue_file_inbound_batch(self, msg_id: str, inbound: InboundMessage) -> bool:
        key = self._pending_key(inbound.chat_id, inbound.user_id)
        should_schedule_flush = False
        expired_batch = None

        with self._thread_lock:
            batch = self._pending_inbound_batches.get(key)
            now = time.time()
            if batch:
                if now - batch["created_at"] <= FEISHU_INBOUND_BATCH_WINDOW_SECONDS:
                    batched_inbound = batch["inbound"]
                    batch["message_ids"].append(msg_id)
                    batch["text_parts"].append(inbound.text)
                    batched_inbound.text = "\n\n".join(part for part in batch["text_parts"] if part)
                    batched_inbound.files.extend(inbound.files)
                    batched_inbound.metadata["batched_message_ids"] = list(batch["message_ids"])
                    logger.info("[Feishu] batched inbound file message: chat_id=%s files=%d", inbound.chat_id, len(batched_inbound.files))
                    return True
                expired_batch = (batch["anchor_message_id"], batch["inbound"], list(batch["message_ids"]))

            self._pending_inbound_batches[key] = {
                "anchor_message_id": msg_id,
                "created_at": now,
                "inbound": inbound,
                "message_ids": [msg_id],
                "text_parts": [inbound.text],
            }
            inbound.metadata["batched_message_ids"] = [msg_id]
            should_schedule_flush = True

        if should_schedule_flush:
            self._schedule_batch_flush(key, msg_id)
        if expired_batch:
            anchor_message_id, expired_inbound, source_message_ids = expired_batch
            self._schedule_prepare_inbound(anchor_message_id, expired_inbound, source_message_ids=source_message_ids)
        return True

    async def _flush_pending_inbound_batch_after(self, key: tuple[str, str], anchor_message_id: str) -> None:
        await asyncio.sleep(FEISHU_INBOUND_BATCH_WINDOW_SECONDS)
        with self._thread_lock:
            batch = self._pending_inbound_batches.get(key)
            if not batch or batch["anchor_message_id"] != anchor_message_id:
                return
            self._pending_inbound_batches.pop(key, None)
            anchor_message_id, inbound, source_message_ids = batch["anchor_message_id"], batch["inbound"], list(batch["message_ids"])
        await self._prepare_inbound(anchor_message_id, inbound, source_message_ids=source_message_ids)

    async def _prepare_inbound(self, msg_id: str, inbound: InboundMessage, *, source_message_ids: list[str] | None = None) -> None:
        inbound = await self._attach_connection_identity(inbound)
        reaction_message_ids = source_message_ids or [msg_id]
        for reaction_message_id in reaction_message_ids:
            reaction_task = asyncio.create_task(self._add_reaction(reaction_message_id, "OK"))
            self._background_tasks.add(reaction_task)
        self._ensure_running_card_started(msg_id, metadata=inbound.metadata)
        await self.bus.publish_inbound(inbound)

    async def _attach_connection_identity(self, inbound: InboundMessage) -> InboundMessage:
        return await attach_connection_identity(
            inbound,
            repo=self._connection_repo,
            provider="feishu",
            workspace_id=inbound.chat_id,
        )

    async def _bind_connection_from_connect_code(self, *, message_id: str, chat_id: str, user_id: str, code: str) -> bool:
        if self._connection_repo is None or not code:
            return False
        state = await self._connection_repo.consume_oauth_state(provider="feishu", state=code)
        if state is None:
            await self._reply_card(message_id, "Feishu connection code is invalid or expired.")
            return True
        if not user_id or not chat_id:
            await self._reply_card(message_id, "Feishu connection could not be completed from this message.")
            return True
        await self._connection_repo.upsert_connection(
            owner_user_id=state["owner_user_id"],
            provider="feishu",
            external_account_id=user_id,
            workspace_id=chat_id,
            metadata={"chat_id": chat_id, "message_id": message_id},
            status="connected",
        )
        await self._reply_card(message_id, "Feishu connected to NovelFactory.")
        return True

    def _on_message(self, event) -> None:
        """Called by lark-oapi when a message is received (runs in lark thread)."""
        try:
            logger.info("[Feishu] raw event received: type=%s", type(event).__name__)
            message = event.event.message
            chat_id = message.chat_id
            msg_id = message.message_id
            sender_id = event.event.sender.sender_id.open_id

            root_id = getattr(message, "root_id", None) or None
            chat_type = getattr(message, "chat_type", None)
            parent_id = self._non_empty_str(getattr(message, "parent_id", None))
            feishu_thread_id = self._non_empty_str(getattr(message, "thread_id", None))

            content = json.loads(message.content)
            files_list = []

            if "text" in content:
                text = content["text"]
            elif "file_key" in content:
                file_key = content.get("file_key")
                if isinstance(file_key, str) and file_key:
                    files_list.append({"file_key": file_key})
                    text = "[file]"
                else:
                    text = ""
            elif "image_key" in content:
                image_key = content.get("image_key")
                if isinstance(image_key, str) and image_key:
                    files_list.append({"image_key": image_key})
                    text = "[image]"
                else:
                    text = ""
            elif "content" in content and isinstance(content["content"], list):
                text_paragraphs = []
                for paragraph in content["content"]:
                    if isinstance(paragraph, list):
                        paragraph_text_parts = []
                        for element in paragraph:
                            if isinstance(element, dict):
                                if element.get("tag") in ("text", "at"):
                                    text_value = element.get("text", "")
                                    if text_value:
                                        paragraph_text_parts.append(text_value)
                                elif element.get("tag") == "img":
                                    image_key = element.get("image_key")
                                    if isinstance(image_key, str) and image_key:
                                        files_list.append({"image_key": image_key})
                                        paragraph_text_parts.append("[image]")
                                elif element.get("tag") in ("file", "media"):
                                    file_key = element.get("file_key")
                                    if isinstance(file_key, str) and file_key:
                                        files_list.append({"file_key": file_key})
                                        paragraph_text_parts.append("[file]")
                        if paragraph_text_parts:
                            text_paragraphs.append(" ".join(paragraph_text_parts))
                text = "\n\n".join(text_paragraphs)
            else:
                text = ""
            text = text.strip()

            if not (text or files_list):
                logger.info("[Feishu] empty text, ignoring message")
                return

            connect_code = self._pending_connect_code(text)
            if connect_code:
                if self._main_loop and self._main_loop.is_running():
                    fut = asyncio.run_coroutine_threadsafe(
                        self._bind_connection_from_connect_code(
                            message_id=msg_id, chat_id=chat_id, user_id=sender_id, code=connect_code,
                        ),
                        self._main_loop,
                    )
                    fut.add_done_callback(lambda f, mid=msg_id: self._log_future_error(f, "bind_connection", mid))
                return

            command_text = strip_leading_mentions(text)
            if _is_feishu_command(command_text):
                msg_type = InboundMessageType.COMMAND
                text = command_text
            else:
                msg_type = InboundMessageType.CHAT

            topic_id, resolved_from_stored_mapping = self._resolve_topic_id(
                chat_id, msg_id, root_id=root_id, parent_id=parent_id, thread_id=feishu_thread_id,
            )
            if chat_type == "p2p" and not resolved_from_stored_mapping:
                topic_id = None
            resolved_from_pending = False
            if msg_type == InboundMessageType.CHAT and not resolved_from_stored_mapping:
                pending = self._consume_pending_clarification(chat_id, sender_id)
                pending_topic_id = self._non_empty_str(pending.get("topic_id")) if pending else None
                if pending_topic_id:
                    topic_id = pending_topic_id
                    resolved_from_pending = True

            source_preview = None
            if self._should_include_source_preview(chat_type=chat_type, root_id=root_id, parent_id=parent_id, thread_id=feishu_thread_id):
                source_preview = self._compact_source_preview(text)

            metadata = {
                "message_id": msg_id,
                "root_id": root_id,
                "parent_id": parent_id,
                "thread_id": feishu_thread_id,
                "topic_id": topic_id,
                "user_id": sender_id,
                RESOLVED_FROM_PENDING_CLARIFICATION_METADATA_KEY: resolved_from_pending,
            }
            if source_preview:
                metadata[SOURCE_PREVIEW_METADATA_KEY] = source_preview

            inbound = self._make_inbound(
                chat_id=chat_id,
                user_id=sender_id,
                text=text,
                msg_type=msg_type,
                thread_ts=msg_id,
                files=files_list,
                metadata=metadata,
            )
            inbound.topic_id = topic_id

            if self._is_batchable_file_inbound(
                msg_type=msg_type, text=text, files=files_list, root_id=root_id, parent_id=parent_id, thread_id=feishu_thread_id,
            ):
                self._queue_file_inbound_batch(msg_id, inbound)
                return

            self._schedule_prepare_inbound(msg_id, inbound)
        except Exception:
            logger.exception("[Feishu] error processing message")