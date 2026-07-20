"""Agent infrastructure utilities — serialization, retry, logging, streaming, quota, circuit breaker."""

from novelfactory.agents.infra.async_retry import async_llm_call_with_retry
from novelfactory.agents.infra.circuit_breaker import (
    circuit_breaker_get_status,
    circuit_breaker_is_open,
    circuit_breaker_record_failure,
    circuit_breaker_record_success,
)
from novelfactory.agents.infra.helpers import (
    extract_ai_message_text,
    extract_fields_from_state,
    make_retry_agent_ainvoke,
)
from novelfactory.agents.infra.llm_cache import LLMResponseCache, get_llm_cache
from novelfactory.agents.infra.logger import get_logger
from novelfactory.agents.infra.quota import get_quota_status, refresh_quota
from novelfactory.agents.infra.retry import (
    TIMEOUT_EXTRACT,
    TIMEOUT_LONG,
    TIMEOUT_SHORT,
    llm_call_with_retry,
)
from novelfactory.agents.infra.serialization import (
    _extract_json_from_text,
    validate_json_output,
)
from novelfactory.agents.infra.stream import (
    StreamWriter,
    cleanup_all_crew_streams,
    cleanup_crew_stream,
    get_crew_stream,
)
from novelfactory.agents.infra.timeout import LLMTimeoutError, with_timeout
from novelfactory.agents.infra.usage import (
    count_tokens,
    count_tokens_dict,
    read_usage_tracking,
    reset_usage_tracking,
)

__all__ = [
    "LLMResponseCache",
    "LLMTimeoutError",
    "StreamWriter",
    "TIMEOUT_EXTRACT",
    "TIMEOUT_LONG",
    "TIMEOUT_SHORT",
    "_extract_json_from_text",
    "circuit_breaker_get_status",
    "circuit_breaker_is_open",
    "circuit_breaker_record_failure",
    "circuit_breaker_record_success",
    "cleanup_all_crew_streams",
    "cleanup_crew_stream",
    "count_tokens",
    "count_tokens_dict",
    "extract_ai_message_text",
    "extract_fields_from_state",
    "make_retry_agent_ainvoke",
    "get_crew_stream",
    "get_logger",
    "get_quota_status",
    "async_llm_call_with_retry",
    "llm_call_with_retry",
    "read_usage_tracking",
    "refresh_quota",
    "reset_usage_tracking",
    "validate_json_output",
    "with_timeout",
]
