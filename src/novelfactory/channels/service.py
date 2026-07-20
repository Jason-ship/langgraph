"""ChannelService — manages the lifecycle of all IM channels.

Migrated from DeerFlow app/channels/service.py.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from novelfactory.channels.base import Channel
from novelfactory.channels.manager import DEFAULT_RECURSION_LIMIT, ChannelManager
from novelfactory.channels.message_bus import MessageBus
from novelfactory.channels.store import ChannelStore

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

# Channel name → import path for lazy loading
_CHANNEL_REGISTRY: dict[str, str] = {
    "feishu": "novelfactory.channels.feishu:FeishuChannel",
}

# Keys that indicate a user has configured credentials for a channel.
_CHANNEL_CREDENTIAL_KEYS: dict[str, list[str]] = {
    "feishu": ["app_id", "app_secret"],
}


def _channel_has_credentials(name: str, channel_config: dict[str, Any]) -> bool:
    cred_keys = _CHANNEL_CREDENTIAL_KEYS.get(name, [])
    return any(
        not isinstance(channel_config.get(key), bool)
        and channel_config.get(key) is not None
        and str(channel_config[key]).strip()
        for key in cred_keys
    )


class ChannelService:
    """Manages the lifecycle of all configured IM channels."""

    def __init__(
        self,
        channels_config: dict[str, Any] | None = None,
        *,
        connection_repo: Any | None = None,
        require_bound_identity: bool = False,
        get_graph: Callable | None = None,
        get_context: Callable | None = None,
    ) -> None:
        self.bus = MessageBus()
        self.store = ChannelStore()
        self._connection_repo = connection_repo
        self.manager = ChannelManager(
            bus=self.bus,
            store=self.store,
            connection_repo=connection_repo,
            require_bound_identity=require_bound_identity,
            get_graph=get_graph,
            get_context=get_context,
        )
        self._channels: dict[str, Any] = {}
        self._config = dict(channels_config or {})
        self._running = False
        self._readiness_locks: dict[str, asyncio.Lock] = {}

    async def start(self) -> None:
        """Start the manager and all enabled channels."""
        if self._running:
            return

        await self.manager.start()
        self._running = True

        ready_status = await self.ensure_ready_channels(attempts=2)
        ready_count = sum(1 for ready in ready_status.values() if ready)
        logger.info("ChannelService started with %d/%d ready channels", ready_count, len(ready_status))

    async def ensure_ready_channels(self, *, attempts: int = 1) -> dict[str, bool]:
        """Start or restart enabled configured channels that are not ready."""
        ready_status: dict[str, bool] = {}
        for name, channel_config in self._config.items():
            if not isinstance(channel_config, dict):
                continue
            if not channel_config.get("enabled", False):
                if _channel_has_credentials(name, channel_config):
                    logger.warning("Channel %s has credentials but is disabled.", name)
                continue
            ready_status[name] = await self.ensure_channel_ready(name, attempts=attempts)
        return ready_status

    async def ensure_channel_ready(self, name: str, config: dict[str, Any] | None = None, *, attempts: int = 1) -> bool:
        """Ensure a single enabled channel is running."""
        if not self._running:
            logger.warning("ChannelService is not running")
            return False

        if config is not None:
            self._config[name] = dict(config)

        lock = self._readiness_locks.setdefault(name, asyncio.Lock())
        async with lock:
            channel_config = self._config.get(name)
            if not channel_config or not isinstance(channel_config, dict):
                return False
            if not channel_config.get("enabled", False):
                return False

            channel = self._channels.get(name)
            if channel is not None and channel.is_running:
                return True

            if channel is not None:
                try:
                    await channel.stop()
                except Exception:
                    logger.exception("Error stopping channel")
                self._channels.pop(name, None)

            for attempt in range(max(1, attempts)):
                if attempt > 0:
                    logger.info("Retrying channel startup")
                if await self._start_channel(name, channel_config):
                    return True
            return False

    async def stop(self) -> None:
        """Stop all channels and the manager."""
        for name, channel in list(self._channels.items()):
            try:
                await channel.stop()
            except Exception:
                logger.exception("Error stopping channel %s", name)
        self._channels.clear()

        await self.manager.stop()
        self._running = False
        logger.info("ChannelService stopped")

    async def restart_channel(self, name: str) -> bool:
        """Restart a specific channel."""
        if name in self._channels:
            try:
                await self._channels[name].stop()
            except Exception:
                logger.exception("Error stopping channel for restart")
            del self._channels[name]

        config = self._config.get(name)
        if not config or not isinstance(config, dict):
            return False
        if not config.get("enabled", False):
            return True
        return await self._start_channel(name, config)

    async def configure_channel(self, name: str, config: dict[str, Any]) -> bool:
        """Apply runtime config for a channel and restart it."""
        self._config[name] = dict(config)
        if not self._running:
            return True
        return await self.restart_channel(name)

    async def remove_channel(self, name: str) -> bool:
        """Remove runtime config for a channel and stop it."""
        self._config.pop(name, None)
        channel = self._channels.pop(name, None)
        if channel is None:
            return True
        try:
            await channel.stop()
            return True
        except Exception:
            logger.exception("Error stopping channel for removal")
            return False

    async def _start_channel(self, name: str, config: dict[str, Any]) -> bool:
        """Instantiate and start a single channel."""
        import_path = _CHANNEL_REGISTRY.get(name)
        if not import_path:
            logger.warning("Unknown channel type: %s", name)
            return False

        try:
            module_path, class_name = import_path.split(":", 1)
            import importlib

            module = importlib.import_module(module_path)
            channel_cls = getattr(module, class_name)
        except Exception:
            logger.exception("Failed to import channel class for %s", name)
            return False

        try:
            config = dict(config)
            config["channel_store"] = self.store
            if self._connection_repo is not None:
                config["connection_repo"] = self._connection_repo
            channel = channel_cls(bus=self.bus, config=config)
            self._channels[name] = channel
            await channel.start()
            if not channel.is_running:
                self._channels.pop(name, None)
                logger.error("Channel %s did not enter running state", name)
                return False
            logger.info("Channel %s started", name)
            return True
        except Exception:
            self._channels.pop(name, None)
            logger.exception("Failed to start channel %s", name)
            return False

    def get_status(self) -> dict[str, Any]:
        """Return status information for all channels."""
        channels_status = {}
        for name in _CHANNEL_REGISTRY:
            config = self._config.get(name, {})
            enabled = isinstance(config, dict) and config.get("enabled", False)
            running = name in self._channels and self._channels[name].is_running
            channels_status[name] = {"enabled": enabled, "running": running}
        return {"service_running": self._running, "channels": channels_status}

    def get_channel(self, name: str) -> Channel | None:
        """Return a running channel instance by name."""
        return self._channels.get(name)

    def is_channel_enabled(self, name: str) -> bool:
        """Return whether a channel is enabled in the live config."""
        config = self._config.get(name)
        if not isinstance(config, dict):
            return False
        return bool(config.get("enabled", False))

    def get_channel_config(self, name: str) -> dict[str, Any] | None:
        """Return a shallow copy of the live config for a channel."""
        config = self._config.get(name)
        if not isinstance(config, dict):
            return None
        return dict(config)


# -- singleton access -------------------------------------------------------

_channel_service: ChannelService | None = None


def get_channel_service() -> ChannelService | None:
    """Get the singleton ChannelService instance (if started)."""
    return _channel_service


async def start_channel_service(
    channels_config: dict[str, Any] | None = None,
    *,
    connection_repo: Any | None = None,
    get_graph: Any = None,
    get_context: Any = None,
) -> ChannelService:
    """Create and start the global ChannelService."""
    global _channel_service
    if _channel_service is not None:
        return _channel_service
    _channel_service = ChannelService(
        channels_config=channels_config,
        connection_repo=connection_repo,
        get_graph=get_graph,
        get_context=get_context,
    )
    await _channel_service.start()
    return _channel_service


async def stop_channel_service() -> None:
    """Stop the global ChannelService."""
    global _channel_service
    if _channel_service is not None:
        await _channel_service.stop()
        _channel_service = None