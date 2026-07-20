"""NovelFactory IM Channels package.

Migrated from DeerFlow channels layer (v2.1.0).
Provides MessageBus, Channel base class, ChannelManager, and FeishuChannel.
"""

from novelfactory.channels.message_bus import (
    InboundMessage,
    InboundMessageType,
    MessageBus,
    OutboundMessage,
    ResolvedAttachment,
)
from novelfactory.channels.base import Channel
from novelfactory.channels.feishu import FeishuChannel
from novelfactory.channels.manager import ChannelManager
from novelfactory.channels.service import (
    ChannelService,
    get_channel_service,
    start_channel_service,
    stop_channel_service,
)
from novelfactory.channels.store import ChannelStore
from novelfactory.channels.run_policy import ChannelRunPolicy, CHANNEL_RUN_POLICY
from novelfactory.channels.feishu_run_policy import register_policy as register_feishu_policy
from novelfactory.channels.commands import KNOWN_CHANNEL_COMMANDS, is_known_channel_command
from novelfactory.channels.runtime_config_store import ChannelRuntimeConfigStore

__all__ = [
    "InboundMessage",
    "InboundMessageType",
    "MessageBus",
    "OutboundMessage",
    "ResolvedAttachment",
    "Channel",
    "FeishuChannel",
    "ChannelManager",
    "ChannelService",
    "ChannelStore",
    "ChannelRunPolicy",
    "CHANNEL_RUN_POLICY",
    "ChannelRuntimeConfigStore",
    "get_channel_service",
    "start_channel_service",
    "stop_channel_service",
    "register_feishu_policy",
    "KNOWN_CHANNEL_COMMANDS",
    "is_known_channel_command",
]