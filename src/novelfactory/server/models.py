# ── Pydantic Models ────────────────────────────────────────────────────────────

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class Assistant(BaseModel):
    """LangGraph assistant entity — represents a configured novel-writing agent."""

    assistant_id: str = "novelfactory"
    name: str = "NovelFactory"
    description: str = "Multi-agent novel writing system"
    model: str = "deepseek-chat"
    config: dict = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ThreadModel(BaseModel):
    """A conversation thread — tracks state across multiple runs."""

    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict = Field(default_factory=dict)


class RunRequest(BaseModel):
    """Request body for creating a graph run — supports both streaming and sync modes."""

    input: dict = Field(default_factory=dict)
    thread_id: str | None = None
    user_id: str = ""
    project_id: str = ""
    lark_config: dict | None = None
    assistant_id: str = "novelfactory"
    stream: bool = True
    config: dict = Field(default_factory=dict)
    command: dict | None = None
    metadata: dict | None = None
    stream_mode: list[str] | None = None
    stream_subgraphs: bool = False
    interrupt_before: list[str] | None = None
    interrupt_after: list[str] | None = None
    checkpoint: str | None = None
    checkpoint_id: str | None = None
    multitask_strategy: str | None = None
    on_completion: str | None = None
    on_disconnect: str | None = "cancel"
    webhook: str | None = None
    feedback_keys: list[str] | None = None
    after_seconds: float | None = None
    if_not_exists: str | None = None


class ThreadState(BaseModel):
    """Thread state — values, next nodes, and interrupts."""

    values: dict = Field(default_factory=dict)
    next: list[str] = Field(default_factory=list)
    interrupts: list[dict] = Field(default_factory=list)
    checkpoint: dict | None = None
    metadata: dict | None = None


class HistoryRequest(BaseModel):
    """Request for listing thread history."""

    limit: int = 10
    before: str | None = None


class StoreItem(BaseModel):
    """Store item."""

    namespace: list[str]
    key: str
    value: dict = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class StoreSearchRequest(BaseModel):
    """Store search request."""

    namespace_prefix: list[str] = Field(default_factory=list)
    filter: dict | None = None
    limit: int = 10
    offset: int = 0


class CronModel(BaseModel):
    """A scheduled cron job — persists a recurring graph run."""

    cron_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    assistant_id: str = "novelfactory"
    schedule: str = "* * * * *"  # cron expression
    input: dict = Field(default_factory=dict)
    status: str = "active"  # active | paused | completed
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_run_at: str | None = None
    next_run_at: str | None = None
    total_runs: int = 0
