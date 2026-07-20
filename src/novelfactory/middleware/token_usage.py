"""Token usage tracking and attribution for agent calls.

Migrated from DeerFlow agents/middlewares/token_usage_middleware.py.

Tracks token consumption per model call, per tool, and provides
aggregated usage stats for observability and billing.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token usage for a single model call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""


@dataclass
class AggregatedUsage:
    """Aggregated token usage across multiple calls."""

    total_prompt: int = 0
    total_completion: int = 0
    total: int = 0
    call_count: int = 0
    by_model: dict[str, TokenUsage] = field(default_factory=dict)
    by_tool: dict[str, TokenUsage] = field(default_factory=dict)


class TokenUsageTracker:
    """Tracks token usage across agent calls.

    Usage:
        tracker = TokenUsageTracker()
        # After each model call:
        tracker.record_call(ai_message, model="deepseek-chat")
        # After each tool call:
        tracker.record_tool(tool_name="chapter_writer", token_usage=usage)
        # Get report:
        report = tracker.get_aggregated()
    """

    def __init__(self) -> None:
        self._calls: list[TokenUsage] = []
        self._tool_usage: dict[str, list[TokenUsage]] = defaultdict(list)
        self._per_thread: dict[str, AggregatedUsage] = defaultdict(AggregatedUsage)

    def record_call(
        self,
        message: AIMessage | None = None,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model: str = "",
        thread_id: str = "",
    ) -> TokenUsage:
        """Record token usage from a model call.

        Extracts token info from AIMessage response_metadata if available,
        or uses explicit values.
        """
        if message is not None:
            metadata = getattr(message, "response_metadata", {}) or {}
            if isinstance(metadata, dict):
                token_usage = metadata.get("token_usage", {}) or metadata.get("usage", {})
                if isinstance(token_usage, dict):
                    prompt_tokens = prompt_tokens or token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0)
                    completion_tokens = completion_tokens or token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0)
                model = model or metadata.get("model_name", "") or metadata.get("model", "")

        total = prompt_tokens + completion_tokens
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            model=model,
        )
        self._calls.append(usage)

        if thread_id:
            agg = self._per_thread[thread_id]
            agg.total_prompt += prompt_tokens
            agg.total_completion += completion_tokens
            agg.total += total
            agg.call_count += 1
            if model:
                if model not in agg.by_model:
                    agg.by_model[model] = TokenUsage(model=model)
                agg.by_model[model].prompt_tokens += prompt_tokens
                agg.by_model[model].completion_tokens += completion_tokens
                agg.by_model[model].total_tokens += total

        return usage

    def record_tool(
        self,
        tool_name: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        thread_id: str = "",
    ) -> None:
        """Record token usage attributed to a specific tool."""
        total = prompt_tokens + completion_tokens
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
        )
        self._tool_usage[tool_name].append(usage)

        if thread_id and tool_name:
            agg = self._per_thread[thread_id]
            if tool_name not in agg.by_tool:
                agg.by_tool[tool_name] = TokenUsage(model=tool_name)
            agg.by_tool[tool_name].prompt_tokens += prompt_tokens
            agg.by_tool[tool_name].completion_tokens += completion_tokens
            agg.by_tool[tool_name].total_tokens += total

    def get_aggregated(self) -> AggregatedUsage:
        """Get aggregated token usage across all calls."""
        agg = AggregatedUsage()
        for usage in self._calls:
            agg.total_prompt += usage.prompt_tokens
            agg.total_completion += usage.completion_tokens
            agg.total += usage.total_tokens
            agg.call_count += 1
            if usage.model:
                if usage.model not in agg.by_model:
                    agg.by_model[usage.model] = TokenUsage(model=usage.model)
                agg.by_model[usage.model].prompt_tokens += usage.prompt_tokens
                agg.by_model[usage.model].completion_tokens += usage.completion_tokens
                agg.by_model[usage.model].total_tokens += usage.total_tokens

        for tool_name, usages in self._tool_usage.items():
            tool_agg = TokenUsage(model=tool_name)
            for u in usages:
                tool_agg.prompt_tokens += u.prompt_tokens
                tool_agg.completion_tokens += u.completion_tokens
                tool_agg.total_tokens += u.total_tokens
            agg.by_tool[tool_name] = tool_agg

        return agg

    def get_thread_usage(self, thread_id: str) -> AggregatedUsage | None:
        """Get aggregated usage for a specific thread."""
        return self._per_thread.get(thread_id)

    def reset(self) -> None:
        """Reset all tracking data."""
        self._calls.clear()
        self._tool_usage.clear()
        self._per_thread.clear()

    def format_report(self) -> str:
        """Format a human-readable usage report."""
        agg = self.get_aggregated()
        lines = [
            f"Token Usage Report ({agg.call_count} calls)",
            f"  Total: {agg.total:,} tokens",
            f"  Prompt: {agg.total_prompt:,} | Completion: {agg.total_completion:,}",
        ]
        if agg.by_model:
            lines.append("  By Model:")
            for model, usage in sorted(agg.by_model.items()):
                lines.append(f"    {model}: {usage.total_tokens:,} tokens ({usage.prompt_tokens:,} + {usage.completion_tokens:,})")
        if agg.by_tool:
            lines.append("  By Tool:")
            for tool, usage in sorted(agg.by_tool.items()):
                lines.append(f"    {tool}: {usage.total_tokens:,} tokens")
        return "\n".join(lines)


__all__ = [
    "TokenUsage",
    "AggregatedUsage",
    "TokenUsageTracker",
]