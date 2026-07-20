"""Detect and break repetitive tool call loops.

Migrated from DeerFlow agents/middlewares/loop_detection_middleware.py.

P0 safety: prevents the agent from calling the same tool with the same
arguments indefinitely until the recursion limit kills the run.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

_DEFAULT_WARN_THRESHOLD = 3
_DEFAULT_HARD_LIMIT = 5
_DEFAULT_WINDOW_SIZE = 20


class LoopDetector:
    """Detects and breaks repetitive tool call loops.

    Tracks tool call hashes in a sliding window and provides warnings
    or hard-stop signals when the same call repeats.
    """

    def __init__(
        self,
        warn_threshold: int = _DEFAULT_WARN_THRESHOLD,
        hard_limit: int = _DEFAULT_HARD_LIMIT,
        window_size: int = _DEFAULT_WINDOW_SIZE,
    ) -> None:
        self._warn_threshold = warn_threshold
        self._hard_limit = hard_limit
        self._window_size = window_size
        # per-thread tracking: {thread_id: deque of hashes}
        self._thread_hashes: dict[str, OrderedDict] = defaultdict(OrderedDict)

    @staticmethod
    def _hash_tool_calls(message: AIMessage) -> str | None:
        """Hash the tool calls (name + args) from an AIMessage."""
        if not message.tool_calls:
            return None
        # Sort by name for deterministic hashing
        sorted_calls = sorted(
            (tc for tc in message.tool_calls if isinstance(tc, dict)),
            key=lambda x: str(x.get("name", "")),
        )
        return hashlib.sha256(
            json.dumps(
                [(tc.get("name", ""), tc.get("args", {})) for tc in sorted_calls],
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()

    def check(self, message: AIMessage, thread_id: str = "default") -> dict[str, Any]:
        """Check an AIMessage for loop detection.

        Returns a dict with:
            - loop_detected: bool
            - warning: str | None (warning message to inject)
            - hard_stop: bool (if True, strip all tool_calls)
        """
        call_hash = self._hash_tool_calls(message)
        if call_hash is None:
            return {"loop_detected": False, "warning": None, "hard_stop": False}

        # Track in sliding window
        thread_data = self._thread_hashes[thread_id]
        thread_data[call_hash] = thread_data.get(call_hash, 0) + 1
        # Evict old entries beyond window
        while len(thread_data) > self._window_size:
            oldest_key = next(iter(thread_data))
            if thread_data[oldest_key] <= 1:
                del thread_data[oldest_key]
            else:
                thread_data[oldest_key] -= 1

        count = thread_data.get(call_hash, 0)

        result = {"loop_detected": count >= self._warn_threshold, "warning": None, "hard_stop": False}

        if count >= self._hard_limit:
            result["hard_stop"] = True
            result["warning"] = (
                f"Hard stop: tool call has been repeated {count} times. "
                "The tool calls have been stripped. Please provide a final text answer."
            )
            logger.warning("[LoopDetector] Hard stop triggered for thread=%s hash=%s count=%d", thread_id, call_hash[:8], count)
        elif count >= self._warn_threshold:
            result["warning"] = (
                f"Warning: this tool call has been repeated {count} times. "
                "If you are stuck in a loop, wrap up and provide a final answer."
            )
            logger.info("[LoopDetector] Warning triggered for thread=%s hash=%s count=%d", thread_id, call_hash[:8], count)

        return result

    def get_stop_reason(self, thread_id: str = "default") -> str | None:
        """Return 'loop_capped' if the thread hit hard limit, else None."""
        thread_data = self._thread_hashes.get(thread_id)
        if thread_data is None:
            return None
        for _, count in thread_data.items():
            if count >= self._hard_limit:
                return "loop_capped"
        return None

    def reset_thread(self, thread_id: str) -> None:
        """Reset tracking for a thread."""
        self._thread_hashes.pop(thread_id, None)


def check_loop(
    message: AIMessage,
    detector: LoopDetector | None = None,
    thread_id: str = "default",
) -> tuple[AIMessage, str | None]:
    """Check an AIMessage for loop detection and return (modified_message, stop_reason).

    If hard_stop is triggered, tool_calls are stripped from the message.
    If warning is triggered, a HumanMessage warning is appended.

    Returns:
        (modified_message, stop_reason)
        stop_reason is "loop_capped" if hard stop, None otherwise.
    """
    if detector is None:
        detector = LoopDetector()

    result = detector.check(message, thread_id=thread_id)

    if result["hard_stop"]:
        # Strip tool calls
        clean = AIMessage(
            content=result["warning"] or "Loop detected. Tool calls stripped.",
            additional_kwargs=message.additional_kwargs,
            response_metadata=message.response_metadata,
            id=message.id,
        )
        return clean, "loop_capped"

    if result["warning"]:
        # Warning is returned, caller should inject it
        logger.info("[LoopDetector] Warning: %s", result["warning"])

    return message, None


__all__ = [
    "LoopDetector",
    "check_loop",
]