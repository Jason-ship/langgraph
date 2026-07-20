"""Browser-facing APIs for IM channel bindings.

Migrated from DeerFlow gateway/routers/channel_connections.py.
"""

from __future__ import annotations

import secrets
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from novelfactory.channels.runtime_config_store import ChannelRuntimeConfigStore
from novelfactory.channels.service import get_channel_service

router = APIRouter(prefix="/api/channels", tags=["channels"])
logger = logging.getLogger(__name__)

_STATE_TTL_SECONDS = 600
_MAX_PENDING_CONNECT_CODES_PER_PROVIDER = 5


class ChannelProviderResponse(BaseModel):
    provider: str
    display_name: str
    enabled: bool
    configured: bool
    connectable: bool
    unavailable_reason: str | None = None
    connection_status: str = "disconnected"
    credential_fields: list[dict[str, str]] = Field(default_factory=list)


class ChannelConnectionResponse(BaseModel):
    id: str
    provider: str
    status: str
    external_account_id: str | None = None
    external_account_name: str | None = None
    workspace_id: str | None = None


class ChannelConnectResponse(BaseModel):
    provider: str
    code: str
    instruction: str
    expires_in: int


class ChannelConfigureRequest(BaseModel):
    config: dict[str, Any]


# -- Provider metadata ------------------------------------------------------

_PROVIDER_META: dict[str, dict[str, Any]] = {
    "feishu": {
        "display_name": "飞书",
        "connectable": True,
        "credential_fields": [
            {"name": "app_id", "label": "App ID", "type": "text", "required": True},
            {"name": "app_secret", "label": "App Secret", "type": "password", "required": True},
        ],
    },
}


def _get_provider_status(provider: str) -> str:
    """Get the connection status for a provider."""
    service = get_channel_service()
    if service is None:
        return "disconnected"
    channel = service.get_channel(provider)
    if channel and channel.is_running:
        return "connected"
    config = service.get_channel_config(provider)
    if config and config.get("enabled", False):
        return "configured"
    return "disconnected"


# -- Routes -----------------------------------------------------------------


@router.get("", response_model=dict)
async def list_channels():
    """List all available channels and their status."""
    service = get_channel_service()
    providers = []
    for provider, meta in _PROVIDER_META.items():
        configured = False
        enabled = False
        if service:
            config = service.get_channel_config(provider)
            if config:
                configured = bool(config.get("app_id"))
                enabled = config.get("enabled", False)

        providers.append(
            ChannelProviderResponse(
                provider=provider,
                display_name=meta["display_name"],
                enabled=enabled,
                configured=configured,
                connectable=meta["connectable"],
                connection_status=_get_provider_status(provider),
                credential_fields=meta["credential_fields"],
            ).model_dump()
        )

    return {"enabled": bool(service and service._running), "providers": providers}


@router.get("/{provider}/connect", response_model=ChannelConnectResponse)
async def connect_channel(provider: str, request: Request):
    """Generate a one-time connect code for binding a channel."""
    if provider not in _PROVIDER_META:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider}")

    service = get_channel_service()
    if service is None or not service._running:
        raise HTTPException(status_code=503, detail="Channel service is not running")

    # Generate a one-time code
    code = secrets.token_urlsafe(16)
    expires_at = datetime.now(UTC) + timedelta(seconds=_STATE_TTL_SECONDS)

    # Store the OAuth state
    owner_user_id = "default"  # Simplified: single-user mode
    repo = service._connection_repo
    if repo is not None:
        await repo.create_oauth_state(
            owner_user_id=owner_user_id,
            provider=provider,
            state=code,
            expires_at=expires_at,
        )

    return ChannelConnectResponse(
        provider=provider,
        code=code,
        instruction=f"Send /connect {code} to the {provider} bot",
        expires_in=_STATE_TTL_SECONDS,
    )


@router.post("/{provider}/configure")
async def configure_channel(provider: str, body: ChannelConfigureRequest):
    """Configure a channel with credentials."""
    if provider not in _PROVIDER_META:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider}")

    service = get_channel_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Channel service is not running")

    config = dict(body.config)
    config["enabled"] = True
    success = await service.configure_channel(provider, config)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to configure channel: {provider}")

    return {"status": "ok", "provider": provider, "message": f"Channel {provider} configured"}


@router.post("/{provider}/disconnect")
async def disconnect_channel(provider: str):
    """Disconnect a channel."""
    service = get_channel_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Channel service is not running")

    # Disable the channel
    config = service.get_channel_config(provider)
    if config:
        config["enabled"] = False
        await service.configure_channel(provider, config)

    # Also stop the channel
    await service.remove_channel(provider)

    return {"status": "ok", "provider": provider, "message": f"Channel {provider} disconnected"}


@router.post("/{provider}/restart")
async def restart_channel(provider: str):
    """Restart a channel."""
    if provider not in _PROVIDER_META:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider}")

    service = get_channel_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Channel service is not running")

    success = await service.restart_channel(provider)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to restart channel: {provider}")

    return {"status": "ok", "provider": provider, "message": f"Channel {provider} restarted"}