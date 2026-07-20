"""Guardrail — tool call authorization protocol and built-in providers.

Migrated from DeerFlow guardrails/provider.py + guardrails/middleware.py.

Provides a pluggable authorization layer for tool calls:
- GuardrailProvider protocol: define your own authorization logic
- BuiltinGuardrailProvider: pattern-based allow/deny rules
- check_tool_guardrail: standalone function for any agent pipeline
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ── Data Structures ────────────────────────────────────────────────────────────


@dataclass
class GuardrailRequest:
    """Context passed to the provider for each tool call."""

    tool_name: str
    tool_input: dict[str, Any]
    agent_id: str | None = None
    thread_id: str | None = None
    is_subagent: bool = False
    timestamp: str = ""
    user_id: str | None = None
    tool_call_id: str | None = None


@dataclass
class GuardrailReason:
    """Structured reason for an allow/deny decision."""

    code: str
    message: str = ""


@dataclass
class GuardrailDecision:
    """Provider's allow/deny verdict."""

    allow: bool
    reasons: list[GuardrailReason] = field(default_factory=list)
    policy_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class GuardrailProvider(Protocol):
    """Contract for pluggable tool-call authorization.

    Any class with these methods works — no base class required.
    """

    name: str

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        """Evaluate whether a tool call should proceed."""
        ...

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        """Async variant."""
        ...


# ── Built-in Provider ──────────────────────────────────────────────────────────


class BuiltinGuardrailProvider:
    """Pattern-based guardrail provider with allow/deny rules.

    Rules are evaluated in order:
    1. If tool_name matches any denied_patterns → DENY
    2. If tool_name matches any allowed_patterns and not denied → ALLOW
    3. Otherwise → ALLOW (default allow)

    Args:
        denied_tools: Set of tool names to always deny.
        max_tool_call_length: Max length of tool call args (chars). Default 0 = no limit.
    """

    name = "builtin"

    def __init__(
        self,
        denied_tools: set[str] | None = None,
        max_tool_call_length: int = 0,
    ) -> None:
        self._denied_tools = frozenset(denied_tools or {})
        self._max_tool_call_length = max_tool_call_length

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        """Evaluate a tool call against the built-in rules."""
        reasons: list[GuardrailReason] = []

        # Rule 1: Denied tool
        if request.tool_name in self._denied_tools:
            reasons.append(
                GuardrailReason(
                    code="builtin.tool_denied",
                    message=f"Tool '{request.tool_name}' is in the denied list.",
                )
            )
            return GuardrailDecision(allow=False, reasons=reasons, policy_id="builtin_deny_list")

        # Rule 2: Tool call args too long
        if self._max_tool_call_length > 0:
            args_str = str(request.tool_input)
            if len(args_str) > self._max_tool_call_length:
                reasons.append(
                    GuardrailReason(
                        code="builtin.tool_call_too_long",
                        message=f"Tool call args ({len(args_str)} chars) exceed max length ({self._max_tool_call_length}).",
                    )
                )
                return GuardrailDecision(allow=False, reasons=reasons, policy_id="builtin_max_length")

        # Default: allow
        return GuardrailDecision(allow=True, reasons=reasons, policy_id="builtin_default")

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        """Async variant — delegates to sync evaluate."""
        return self.evaluate(request)


# ── Standalone Check Function ──────────────────────────────────────────────────


def check_tool_guardrail(
    tool_name: str,
    tool_input: dict[str, Any],
    provider: GuardrailProvider | None = None,
    thread_id: str | None = None,
) -> GuardrailDecision:
    """Check a single tool call against a guardrail provider.

    This is a standalone function that can be used in any agent pipeline
    without needing the full LangChain middleware system.

    Args:
        tool_name: The name of the tool being called.
        tool_input: The arguments to the tool.
        provider: The guardrail provider to use. Defaults to BuiltinGuardrailProvider.
        thread_id: Optional thread ID for context.

    Returns:
        GuardrailDecision with allow/deny verdict.
    """
    if provider is None:
        provider = BuiltinGuardrailProvider()

    request = GuardrailRequest(
        tool_name=tool_name,
        tool_input=tool_input,
        thread_id=thread_id,
    )

    try:
        return provider.evaluate(request)
    except Exception as exc:
        logger.exception("[guardrail] Provider %s raised during evaluation", provider.name)
        return GuardrailDecision(
            allow=False,
            reasons=[GuardrailReason(code="builtin.provider_error", message=str(exc))],
            policy_id="builtin_error",
        )


__all__ = [
    "GuardrailRequest",
    "GuardrailReason",
    "GuardrailDecision",
    "GuardrailProvider",
    "BuiltinGuardrailProvider",
    "check_tool_guardrail",
]