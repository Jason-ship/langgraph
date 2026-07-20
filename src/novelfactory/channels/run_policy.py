"""Per-channel run policy registry.

Migrated from DeerFlow app/channels/run_policy.py.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from novelfactory.channels.message_bus import InboundMessage


@dataclass(frozen=True, slots=True)
class ChannelRunPolicy:
    """Per-channel knobs applied by ChannelManager.

    Attributes:
        is_interactive: When False, disable_clarification is set.
        default_recursion_limit: When set, raises recursion_limit.
        credentials_provider: Optional async hook that mutates run_context.
        requires_bound_identity: When False, skips bound-identity gate.
        fire_and_forget: When True, uses fire-and-forget run mode.
        serialize_thread_runs: When True, serializes same-thread inbound turns.
    """

    is_interactive: bool = True
    default_recursion_limit: int | None = None
    credentials_provider: Callable[[InboundMessage, dict[str, Any]], Awaitable[None]] | None = None
    requires_bound_identity: bool = True
    fire_and_forget: bool = False
    serialize_thread_runs: bool = False


# Channel name → policy
CHANNEL_RUN_POLICY: dict[str, ChannelRunPolicy] = {}