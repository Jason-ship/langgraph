"""Suppress tool execution when the provider safety-terminated the response.

Migrated from DeerFlow agents/middlewares/safety_finish_reason_middleware.py.

Some providers (DeepSeek finish_reason='content_filter', etc.) can stop
generation mid-stream while still returning partially-formed tool_calls.
This middleware strips those tool_calls when a safety termination is detected.
"""

from __future__ import annotations

import logging
from typing import Any, override

from langchain_core.messages import AIMessage

from novelfactory.middleware.safety_detectors import (
    SafetyTermination,
    SafetyTerminationDetector,
    default_detectors,
)

logger = logging.getLogger(__name__)

_USER_FACING_MESSAGE = (
    "The model provider stopped this response with a safety-related signal "
    "({reason_field}={reason_value!r}, detector={detector!r}). Any tool "
    "calls produced in this turn were suppressed because their arguments "
    "may be truncated and unsafe to execute. Please rephrase the request."
)


def _clone_ai_message_without_tool_calls(message: AIMessage) -> AIMessage:
    """Create a copy of the AIMessage with tool_calls stripped."""
    safety_termination = getattr(message, "additional_kwargs", {}).get("safety_termination")
    return AIMessage(
        content=_USER_FACING_MESSAGE.format(
            reason_field=safety_termination.get("reason_field", "?"),
            reason_value=safety_termination.get("reason_value", "?"),
            detector=safety_termination.get("detector", "?"),
        ) if safety_termination else message.content,
        additional_kwargs={**message.additional_kwargs, "safety_termination": safety_termination}
        if safety_termination
        else message.additional_kwargs,
        response_metadata=message.response_metadata,
        id=message.id,
    )


def check_safety_termination(message: AIMessage, detectors: list[SafetyTerminationDetector] | None = None) -> AIMessage:
    """Check an AIMessage for safety termination signals and strip tool_calls if found.

    This is a standalone function that can be used in any agent pipeline,
    not just the LangChain middleware system.

    Args:
        message: The AIMessage to check.
        detectors: List of safety termination detectors. Uses defaults if None.

    Returns:
        The (possibly modified) AIMessage with tool_calls stripped if safety termination detected.
    """
    if detectors is None:
        detectors = default_detectors()

    for detector in detectors:
        try:
            result = detector.detect(message)
        except Exception:
            logger.exception("[safety] Detector %s raised during detection", detector.name)
            continue

        if result is not None:
            logger.warning(
                "[safety] Detected provider safety termination: detector=%s, field=%s, value=%s",
                result.detector,
                result.reason_field,
                result.reason_value,
            )
            if message.tool_calls:
                # Create a clean copy without tool_calls
                clean = AIMessage(
                    content=_USER_FACING_MESSAGE.format(
                        reason_field=result.reason_field,
                        reason_value=result.reason_value,
                        detector=result.detector,
                    ),
                    additional_kwargs={
                        **message.additional_kwargs,
                        "safety_termination": {
                            "detector": result.detector,
                            "reason_field": result.reason_field,
                            "reason_value": result.reason_value,
                            "extras": result.extras,
                        },
                    },
                    response_metadata=message.response_metadata,
                    id=message.id,
                )
                return clean
            break

    return message


__all__ = [
    "check_safety_termination",
    "SafetyTermination",
    "SafetyTerminationDetector",
    "default_detectors",
]