"""Subtask Tracker — generates SSE custom events for sub-task tracking.

DeerFlow frontend expects task_started and task_running events to track
sub-agent progress in the chat UI.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class SubtaskTracker:
    """Subtask tracker — generates SSE custom events for sub-task progress.

    Usage:
        tracker = SubtaskTracker(run_id, thread_id)
        # In a streaming node:
        yield tracker.start_subtask("id", "writing_agent", "Generating chapter...")
        # ... do work ...
        yield tracker.update_subtask("id", "completed", content=result)
    """

    def __init__(self, run_id: str, thread_id: str) -> None:
        self._run_id = run_id
        self._thread_id = thread_id
        self._active_subtasks: dict[str, dict[str, Any]] = {}

    def start_subtask(
        self,
        subtask_id: str,
        agent_type: str,
        description: str,
        model_name: str = "",
        prompt: str = "",
    ) -> dict:
        """Generate a task_started SSE event."""
        event = {
            "type": "task_started",
            "subtask_id": subtask_id,
            "subagent_type": agent_type,
            "description": description,
            "model_name": model_name,
            "prompt": prompt[:500],
        }
        self._active_subtasks[subtask_id] = {
            "status": "in_progress",
            "started_at": datetime.now().isoformat(),
        }
        logger.info(
            "[SubtaskTracker] Started: %s (%s) - %s",
            subtask_id, agent_type, description[:50],
        )
        return self._format_sse(event)

    def update_subtask(
        self,
        subtask_id: str,
        status: str = "in_progress",
        content: str = "",
        usage: dict | None = None,
    ) -> dict | None:
        """Generate a task_running SSE event."""
        task = self._active_subtasks.get(subtask_id)
        if not task:
            logger.warning("[SubtaskTracker] Unknown subtask: %s", subtask_id)
            return None

        task["status"] = status
        event: dict[str, Any] = {
            "type": "task_running",
            "subtask_id": subtask_id,
            "status": status,
            "latest_message": {
                "role": "assistant",
                "content": content[:1000] if content else "",
            },
        }
        if usage:
            event["usage"] = usage
        if status == "completed":
            task["completed_at"] = datetime.now().isoformat()
            logger.info(
                "[SubtaskTracker] Completed: %s (tokens: %s)",
                subtask_id, usage.get("total_tokens", "N/A") if usage else "N/A",
            )

        return self._format_sse(event)

    def _format_sse(self, data: dict) -> dict:
        """Format as an SSE custom event matching DeerFlow's expected format."""
        return {
            "event": "custom",
            "data": json.dumps(data, default=str),
        }