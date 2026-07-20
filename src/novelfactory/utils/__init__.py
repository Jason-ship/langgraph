"""NovelFactory utility modules.

Migrated utility modules from DeerFlow:
- reflection: Dynamic class/variable resolution from string paths
- time_utils: ISO 8601 timestamp helpers
- message_utils: Message content extraction and conversion
- serialization: Canonical LangChain/LangGraph object serialization
- user_context: Task-local user context via ContextVar
"""

from novelfactory.utils.message_utils import message_content_to_text, message_to_text
from novelfactory.utils.reflection import resolve_class, resolve_variable
from novelfactory.utils.serialization import serialize_channel_values, serialize_lc_object
from novelfactory.utils.sse import format_sse, format_sse_event
from novelfactory.utils.time_utils import coerce_iso, now_iso
from novelfactory.utils.user_context import (
    CurrentUser,
    get_current_user,
    get_effective_user_id,
    require_current_user,
    set_current_user,
)

__all__ = [
    "resolve_variable",
    "resolve_class",
    "now_iso",
    "coerce_iso",
    "message_content_to_text",
    "message_to_text",
    "serialize_lc_object",
    "serialize_channel_values",
    "format_sse",
    "format_sse_event",
    "CurrentUser",
    "set_current_user",
    "get_current_user",
    "require_current_user",
    "get_effective_user_id",
]
