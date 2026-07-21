"""ChannelManager — consumes inbound messages and dispatches them to the NovelFactory agent.

Migrated from DeerFlow app/channels/manager.py.
Adapted for embedded LangGraph runtime (direct graph.ainvoke/astream_events calls).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from novelfactory.channels import feishu_run_policy as _feishu_run_policy  # noqa: F401
from novelfactory.channels.commands import KNOWN_CHANNEL_COMMANDS
from novelfactory.channels.message_bus import (
    PENDING_CLARIFICATION_METADATA_KEY,
    InboundMessage,
    InboundMessageType,
    MessageBus,
    OutboundMessage,
    ResolvedAttachment,
)
from novelfactory.channels.run_policy import CHANNEL_RUN_POLICY, ChannelRunPolicy
from novelfactory.channels.store import ChannelStore

logger = logging.getLogger(__name__)

# 与 config.constants 中的根图 5000 / 子图 200 不同，这是 ChannelManager
# 用于 IM 对话的递归上限，IM 交互需要更短的递归以避免长时间阻塞。
DEFAULT_RECURSION_LIMIT = 100
STREAM_UPDATE_MIN_INTERVAL_SECONDS = 1.0
STREAM_UPDATE_MIN_CHARS = 60
INBOUND_DEDUPE_TTL_SECONDS = 10 * 60
INBOUND_DEDUPE_MAX_ENTRIES = 4096
INBOUND_DEDUPE_METADATA_KEYS = ("event_id", "message_id", "msg_id")
CHAT_SCOPED_WORKSPACE_CHANNELS = frozenset({"feishu", "telegram"})

THREAD_BUSY_MESSAGE = "This conversation is already processing another request. Please wait for it to finish and try again."
BOUND_IDENTITY_REQUIRED_MESSAGE = "Connect this channel from Settings, then send your message again."

_METADATA_DROP_KEYS = frozenset({"raw_message", "ref_msg"})


def _slim_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in meta.items() if k not in _METADATA_DROP_KEYS}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _merge_dicts(*layers: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for layer in layers:
        if isinstance(layer, Mapping):
            merged.update(layer)
    return merged


@dataclass(frozen=True, slots=True)
class _BoundIdentityRejection:
    message: str = BOUND_IDENTITY_REQUIRED_MESSAGE
    outbound_connection_id: str | None = None
    outbound_owner_user_id: str | None = None


@dataclass(slots=True)
class _SerializedThreadRunState:
    lock: asyncio.Lock
    waiters: int = 0


def _extract_response_text(result: dict | list) -> str:
    """Extract the last AI message text from a run result."""
    if isinstance(result, list):
        messages = result
    elif isinstance(result, dict):
        messages = result.get("messages", [])
    else:
        return ""

    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        msg_type = msg.get("type")
        if msg_type == "human":
            break
        if msg_type == "ai":
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                return content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        parts.append(block)
                text = "".join(parts)
                if text:
                    return text
    return ""


def _human_input_message(content: str) -> dict[str, Any]:
    return {"role": "human", "content": content}


def _unknown_command_reply(command: str | None = None) -> str:
    available = " | ".join(sorted(KNOWN_CHANNEL_COMMANDS))
    if command:
        return f"Unknown command: /{command}. Available commands: {available}"
    return f"Unknown command. Available commands: {available}"


class ChannelManager:
    """Core dispatcher that bridges IM channels to the NovelFactory agent.

    Reads from the MessageBus inbound queue, creates/reuses threads,
    runs the agent via direct graph.ainvoke/astream_events calls,
    and publishes outbound responses back through the bus.
    """

    def __init__(
        self,
        bus: MessageBus,
        store: ChannelStore,
        *,
        max_concurrency: int = 5,
        default_recursion_limit: int = DEFAULT_RECURSION_LIMIT,
        get_graph: Any = None,
        get_context: Any = None,
        connection_repo: Any | None = None,
        require_bound_identity: bool = False,
    ) -> None:
        self.bus = bus
        self.store = store
        self._max_concurrency = max_concurrency
        self._default_recursion_limit = default_recursion_limit
        self._get_graph = get_graph
        self._get_context = get_context
        self._connection_repo = connection_repo
        self._require_bound_identity = require_bound_identity

        self._thread_create_locks: dict[tuple[str, str, str | None], asyncio.Lock] = {}
        self._serialized_thread_runs: dict[tuple[str, str], _SerializedThreadRunState] = {}
        self._semaphore: asyncio.Semaphore | None = None
        self._running = False
        self._task: asyncio.Task | None = None
        self._recent_inbound_events: OrderedDict[tuple[str, str, str, str], float] = OrderedDict()

    @staticmethod
    def _channel_supports_streaming(channel_name: str) -> bool:
        from novelfactory.channels.service import get_channel_service

        service = get_channel_service()
        if service:
            channel = service.get_channel(channel_name)
            if channel is not None:
                return channel.supports_streaming
        return False

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Start the dispatch loop."""
        if self._running:
            return
        self._running = True
        self._semaphore = asyncio.Semaphore(self._max_concurrency)
        self._task = asyncio.create_task(self._dispatch_loop())
        logger.info("ChannelManager started (max_concurrency=%d)", self._max_concurrency)

    async def stop(self) -> None:
        """Stop the dispatch loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("ChannelManager stopped")

    # -- dispatch loop -----------------------------------------------------

    async def _dispatch_loop(self) -> None:
        logger.info("[Manager] dispatch loop started, waiting for inbound messages")
        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.get_inbound(), timeout=1.0)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if self._is_duplicate_inbound(msg):
                continue
            logger.info(
                "[Manager] received inbound: channel=%s, chat_id=%s, type=%s, text_len=%d",
                msg.channel_name,
                msg.chat_id,
                msg.msg_type.value,
                len(msg.text or ""),
            )
            task = asyncio.create_task(self._handle_message(msg))
            task.add_done_callback(self._log_task_error)

    @staticmethod
    def _log_task_error(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error("[Manager] unhandled error in message task: %s", exc, exc_info=exc)

    # -- deduplication -----------------------------------------------------

    @staticmethod
    def _inbound_dedupe_key(msg: InboundMessage) -> tuple[str, str, str, str] | None:
        metadata = msg.metadata or {}
        message_id = None
        for key in INBOUND_DEDUPE_METADATA_KEYS:
            value = metadata.get(key)
            if value:
                message_id = str(value)
                break
        if message_id is None:
            return None

        workspace_id = msg.workspace_id or metadata.get("workspace_id") or metadata.get("team_id") or metadata.get("conversation_id")
        if not workspace_id and msg.channel_name in CHAT_SCOPED_WORKSPACE_CHANNELS:
            workspace_id = msg.chat_id or None
        if not workspace_id:
            return None
        return (msg.channel_name, str(workspace_id), msg.chat_id, message_id)

    def _is_duplicate_inbound(self, msg: InboundMessage) -> bool:
        key = self._inbound_dedupe_key(msg)
        if key is None:
            return False

        now = time.monotonic()
        while self._recent_inbound_events:
            _, oldest_at = next(iter(self._recent_inbound_events.items()))
            if now - oldest_at > INBOUND_DEDUPE_TTL_SECONDS:
                self._recent_inbound_events.popitem(last=False)
            else:
                break
        while len(self._recent_inbound_events) > INBOUND_DEDUPE_MAX_ENTRIES:
            self._recent_inbound_events.popitem(last=False)

        if key in self._recent_inbound_events:
            logger.info("[Manager] duplicate inbound ignored: channel=%s, chat_id=%s", msg.channel_name, msg.chat_id)
            return True

        self._recent_inbound_events[key] = now
        return False

    # -- message handling --------------------------------------------------

    async def _handle_message(self, msg: InboundMessage) -> None:
        try:
            bound_identity_rejection = None
            if msg.msg_type != InboundMessageType.COMMAND:
                bound_identity_rejection = await self._get_bound_identity_rejection(msg)
            if bound_identity_rejection is not None:
                await self._reject_unbound_channel_message(msg, bound_identity_rejection=bound_identity_rejection)
                return

            async with self._semaphore:
                if msg.msg_type == InboundMessageType.COMMAND:
                    await self._handle_command(msg)
                else:
                    await self._handle_chat(msg, bound_identity_checked=True)
        except Exception:
            logger.exception("Error handling message from %s (chat=%s)", msg.channel_name, msg.chat_id)
            await self._send_error(msg, "An internal error occurred. Please try again.")

    # -- chat handling -----------------------------------------------------

    async def _get_bound_identity_rejection(self, msg: InboundMessage) -> _BoundIdentityRejection | None:
        if not self._require_bound_identity:
            return None
        policy = CHANNEL_RUN_POLICY.get(msg.channel_name)
        if policy is not None and not policy.requires_bound_identity:
            return None

        has_connection = bool(msg.connection_id)
        has_owner = bool(msg.owner_user_id)
        if not (has_connection and has_owner):
            return _BoundIdentityRejection()
        if self._connection_repo is None:
            return _BoundIdentityRejection(message="Channel connection verification is temporarily unavailable.")

        connection = await self._connection_repo.find_connection_by_external_identity(
            provider=msg.channel_name,
            external_account_id=msg.user_id,
            workspace_id=msg.workspace_id or None,
        )
        if connection is None:
            return _BoundIdentityRejection()

        connection_id = connection.get("id")
        owner_user_id = connection.get("owner_user_id")
        if connection_id == msg.connection_id and owner_user_id == msg.owner_user_id:
            return None
        return _BoundIdentityRejection(outbound_connection_id=connection_id, outbound_owner_user_id=owner_user_id)

    async def _reject_unbound_channel_message(self, msg: InboundMessage, *, bound_identity_rejection: _BoundIdentityRejection) -> None:
        outbound = OutboundMessage(
            channel_name=msg.channel_name,
            chat_id=msg.chat_id,
            thread_id="",
            text=bound_identity_rejection.message,
            thread_ts=msg.thread_ts,
            connection_id=bound_identity_rejection.outbound_connection_id,
            owner_user_id=bound_identity_rejection.outbound_owner_user_id,
            metadata=_slim_metadata(msg.metadata),
        )
        await self.bus.publish_outbound(outbound)

    async def _lookup_thread_id(self, msg: InboundMessage) -> str | None:
        if msg.connection_id and self._connection_repo is not None:
            return await self._connection_repo.get_thread_id(
                msg.connection_id,
                msg.chat_id,
                msg.topic_id,
            )
        return self.store.get_thread_id(msg.channel_name, msg.chat_id, topic_id=msg.topic_id)

    async def _store_thread_id(self, msg: InboundMessage, thread_id: str) -> None:
        if msg.connection_id and msg.owner_user_id and self._connection_repo is not None:
            await self._connection_repo.set_thread_id(
                connection_id=msg.connection_id,
                owner_user_id=msg.owner_user_id,
                provider=msg.channel_name,
                external_conversation_id=msg.chat_id,
                external_topic_id=msg.topic_id,
                thread_id=thread_id,
            )
            return

        self.store.set_thread_id(
            msg.channel_name,
            msg.chat_id,
            thread_id,
            topic_id=msg.topic_id,
            user_id=msg.user_id,
        )

    async def _create_thread(self, msg: InboundMessage) -> str:
        """Create a new thread and store the mapping."""
        import uuid

        thread_id = str(uuid.uuid4())
        await self._store_thread_id(msg, thread_id)
        logger.info("[Manager] new thread created: thread_id=%s for chat_id=%s", thread_id, msg.chat_id)
        return thread_id

    async def _get_or_create_thread(self, msg: InboundMessage) -> tuple[str, bool]:
        """Return ``(thread_id, created)``, creating a thread only if needed."""
        thread_id = await self._lookup_thread_id(msg)
        if thread_id:
            return thread_id, False

        key = (msg.channel_name, msg.chat_id, msg.topic_id)
        lock = self._thread_create_locks.setdefault(key, asyncio.Lock())
        try:
            async with lock:
                thread_id = await self._lookup_thread_id(msg)
                if thread_id:
                    return thread_id, False
                return await self._create_thread(msg), True
        finally:
            self._thread_create_locks.pop(key, None)

    async def _handle_chat(self, msg: InboundMessage, *, bound_identity_checked: bool = False) -> None:
        bound_identity_rejection = None if bound_identity_checked else await self._get_bound_identity_rejection(msg)
        if bound_identity_rejection is not None:
            await self._reject_unbound_channel_message(msg, bound_identity_rejection=bound_identity_rejection)
            return

        thread_id, created = await self._get_or_create_thread(msg)
        if not created:
            logger.info("[Manager] reusing thread: thread_id=%s for topic_id=%s", thread_id, msg.topic_id)

        serial_state, queued = self._begin_serialized_thread_run(
            channel_name=msg.channel_name,
            thread_id=thread_id,
        )
        serial_lock_acquired = False
        try:
            if queued:
                await self._publish_progress_update(msg, thread_id, "Queued behind another request...")
            if serial_state is not None:
                await serial_state.lock.acquire()
                serial_lock_acquired = True
            if queued:
                await self._publish_progress_update(msg, thread_id, "thinking...")
            await self._handle_chat_on_thread(msg, thread_id)
        finally:
            self._finish_serialized_thread_run(
                channel_name=msg.channel_name,
                thread_id=thread_id,
                state=serial_state,
                lock_acquired=serial_lock_acquired,
            )

    def _begin_serialized_thread_run(self, *, channel_name: str, thread_id: str) -> tuple[_SerializedThreadRunState | None, bool]:
        policy = CHANNEL_RUN_POLICY.get(channel_name)
        if policy is None or not policy.serialize_thread_runs:
            return None, False

        key = (channel_name, thread_id)
        state = self._serialized_thread_runs.get(key)
        if state is None:
            state = _SerializedThreadRunState(lock=asyncio.Lock())
            self._serialized_thread_runs[key] = state
        queued = state.lock.locked()
        state.waiters += 1
        return state, queued

    def _finish_serialized_thread_run(self, *, channel_name: str, thread_id: str, state: _SerializedThreadRunState | None, lock_acquired: bool) -> None:
        if state is None:
            return
        if lock_acquired:
            state.lock.release()
        state.waiters -= 1
        if state.waiters == 0 and not state.lock.locked():
            self._serialized_thread_runs.pop((channel_name, thread_id), None)

    async def _publish_progress_update(self, msg: InboundMessage, thread_id: str, text: str) -> None:
        await self.bus.publish_outbound(
            OutboundMessage(
                channel_name=msg.channel_name,
                chat_id=msg.chat_id,
                thread_id=thread_id,
                text=text,
                is_final=False,
                thread_ts=msg.thread_ts,
                connection_id=msg.connection_id,
                owner_user_id=msg.owner_user_id,
                metadata=_response_metadata(msg.metadata),
            )
        )

    async def _handle_chat_on_thread(self, msg: InboundMessage, thread_id: str) -> None:
        human_message = _human_input_message(msg.text)

        if self._channel_supports_streaming(msg.channel_name):
            await self._handle_streaming_chat(msg, thread_id, human_message)
            return

        # Non-streaming path: run agent synchronously
        graph = self._get_graph() if self._get_graph else None
        if graph is None:
            await self._send_error(msg, "Agent not available. Please try again later.")
            return

        try:
            context = self._get_context(thread_id) if self._get_context else {"thread_id": thread_id, "user_id": msg.owner_user_id or msg.user_id}
            result = await graph.ainvoke(
                {"messages": [human_message]},
                config={"configurable": {"thread_id": thread_id}, "recursion_limit": self._default_recursion_limit},
                context=context,
            )
        except Exception as exc:
            if "already running" in str(exc).lower():
                await self._send_error(msg, THREAD_BUSY_MESSAGE)
                return
            raise

        response_text = _extract_response_text(result)

        if not response_text:
            response_text = "(No response from agent)"

        outbound = OutboundMessage(
            channel_name=msg.channel_name,
            chat_id=msg.chat_id,
            thread_id=thread_id,
            text=response_text,
            thread_ts=msg.thread_ts,
            connection_id=msg.connection_id,
            owner_user_id=msg.owner_user_id,
            metadata=_response_metadata(msg.metadata),
        )
        await self.bus.publish_outbound(outbound)

    async def _handle_streaming_chat(self, msg: InboundMessage, thread_id: str, human_message: dict[str, Any]) -> None:
        graph = self._get_graph() if self._get_graph else None
        if graph is None:
            await self._send_error(msg, "Agent not available.")
            return

        last_values: dict[str, Any] | None = None
        latest_text = ""
        last_published_text = ""
        last_published_len = 0
        last_publish_at = 0.0
        stream_error: BaseException | None = None

        try:
            context = self._get_context(thread_id) if self._get_context else {"thread_id": thread_id, "user_id": msg.owner_user_id or msg.user_id}
            async for event in graph.astream_events(
                {"messages": [human_message]},
                config={"configurable": {"thread_id": thread_id}, "recursion_limit": self._default_recursion_limit},
                version="v2",
                context=context,
            ):
                event_kind = event.get("event", "")
                data = event.get("data", {})

                if event_kind in ("messages", "messages-tuple") and isinstance(data, dict):
                    content = data.get("content", "")
                    if isinstance(content, str) and content:
                        latest_text = latest_text + content
                elif event_kind == "values" and isinstance(data, dict):
                    last_values = data

                if not latest_text or latest_text == last_published_text:
                    continue

                now = time.monotonic()
                new_chars = len(latest_text) - last_published_len
                if last_published_text:
                    if now - last_publish_at < STREAM_UPDATE_MIN_INTERVAL_SECONDS and new_chars < STREAM_UPDATE_MIN_CHARS:
                        continue

                display_text = latest_text + " ▉"
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel_name=msg.channel_name,
                        chat_id=msg.chat_id,
                        thread_id=thread_id,
                        text=display_text,
                        is_final=False,
                        thread_ts=msg.thread_ts,
                        connection_id=msg.connection_id,
                        owner_user_id=msg.owner_user_id,
                        metadata=_response_metadata(msg.metadata),
                    )
                )
                last_published_text = latest_text
                last_published_len = len(latest_text)
                last_publish_at = now
        except Exception as exc:
            stream_error = exc
            logger.exception("[Manager] streaming error: thread_id=%s", thread_id)

        # Final outbound
        response_text = latest_text or "(No response from agent)"
        await self.bus.publish_outbound(
            OutboundMessage(
                channel_name=msg.channel_name,
                chat_id=msg.chat_id,
                thread_id=thread_id,
                text=response_text,
                is_final=True,
                thread_ts=msg.thread_ts,
                connection_id=msg.connection_id,
                owner_user_id=msg.owner_user_id,
                metadata=_response_metadata(msg.metadata),
            )
        )

    # -- command handling --------------------------------------------------

    async def _handle_command(self, msg: InboundMessage) -> None:
        bound_identity_rejection = await self._get_bound_identity_rejection(msg)
        if bound_identity_rejection is not None:
            await self._reject_unbound_channel_message(msg, bound_identity_rejection=bound_identity_rejection)
            return

        text = msg.text.strip()
        parts = text.split(maxsplit=1)
        command = parts[0].lower().removeprefix("/") if parts else None
        reply: str | None = None

        if command == "new":
            await self._create_thread(msg)
            reply = "New conversation started."
        elif command == "status":
            thread_id = await self._lookup_thread_id(msg)
            reply = f"Active thread: {thread_id}" if thread_id else "No active conversation."
        elif command == "help":
            reply = (
                "Available commands:\n"
                "/new — Start a new conversation\n"
                "/status — Show current thread info\n"
                "/help — Show this help"
            )
        else:
            reply = _unknown_command_reply(command)

        outbound = OutboundMessage(
            channel_name=msg.channel_name,
            chat_id=msg.chat_id,
            thread_id=await self._lookup_thread_id(msg) or "",
            text=reply,
            thread_ts=msg.thread_ts,
            connection_id=msg.connection_id,
            owner_user_id=msg.owner_user_id,
            metadata=_slim_metadata(msg.metadata),
        )
        await self.bus.publish_outbound(outbound)

    # -- error helper ------------------------------------------------------

    async def _send_error(self, msg: InboundMessage, error_text: str) -> None:
        outbound = OutboundMessage(
            channel_name=msg.channel_name,
            chat_id=msg.chat_id,
            thread_id=await self._lookup_thread_id(msg) or "",
            text=error_text,
            thread_ts=msg.thread_ts,
            connection_id=msg.connection_id,
            owner_user_id=msg.owner_user_id,
            metadata=_slim_metadata(msg.metadata),
        )
        await self.bus.publish_outbound(outbound)


def _response_metadata(base_metadata: dict[str, Any], *, pending_clarification: bool = False) -> dict[str, Any]:
    metadata = _slim_metadata(base_metadata)
    if pending_clarification:
        metadata[PENDING_CLARIFICATION_METADATA_KEY] = True
    return metadata