"""State definitions for NovelFactory LangGraph Multi-Agent Architecture.

本模块仅包含 TypedDict schema 定义（QuotaInfo / NovelFactoryState）。
自定义 reducer 和工具函数已拆分至 ``novelfactory.state.reducers``。

为保持向后兼容，reducer 仍在此模块 re-export。
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, NotRequired, TypedDict

from langgraph.graph.message import add_messages

from novelfactory.state.crew_state import ProjectContext
from novelfactory.state.managed_values import ChapterProgress, PhaseStatus
from novelfactory.state.reducers import (
    _add_chapters_compressed,
    _add_usage,
    _chapter_key,
    _last_value,
    compress_completed_chapters,
)

__all__ = [
    "NovelFactoryState",
    "QuotaInfo",
    "compress_completed_chapters",
    "_last_value",
    "_add_usage",
    "_chapter_key",
    "_add_chapters_compressed",
]


class QuotaInfo(TypedDict, total=False):
    """MiniMax API quota information."""

    remaining_tokens: int
    total_tokens: int
    used_tokens: int
    remaining_pct: float
    reset_time: int
    seconds_until_reset: int
    is_exhausted: bool
    is_critical: bool
    model_quotas: list[dict]
    raw: str


class NovelFactoryState(TypedDict):
    """Global state shared across all Crews.

    This is the root state for the entire application. Each project
    is isolated by thread_id. All Crews read from and write to this state.

    Design principles:
    - Persistent fields (survive checkpoint): all explicitly declared fields
    - Ephemeral fields (computed each run): ManagedValue or _temp_ prefix
    - Scalar fields with multi-source writes: Annotated[T, _last_value]
    - Accumulated fields: Annotated[list, operator.add / add_messages / _add_usage]
    """

    # ═══════════════════════════════════════════════════════════════════════════
    #  Identity & Project
    # ═══════════════════════════════════════════════════════════════════════════

    # NOTE: TypedDict does NOT support field defaults. The `= ""` syntax below is
    # invalid in Python 3.11/3.12 TypedDict and silently ignored at runtime.
    # Default values are handled in node functions via dict.get("field", default).
    thread_id: NotRequired[Annotated[str, _last_value]]
    project_context: NotRequired[ProjectContext]

    # ═══════════════════════════════════════════════════════════════════════════
    #  User Input
    # ═══════════════════════════════════════════════════════════════════════════

    seed_idea: NotRequired[Annotated[str, _last_value]]
    genre: NotRequired[Annotated[str, _last_value]]
    project_name: NotRequired[Annotated[str, _last_value]]
    target_chapters: NotRequired[Annotated[int, _last_value]]
    current_chapter: NotRequired[Annotated[int, _last_value]]

    # ═══════════════════════════════════════════════════════════════════════════
    #  Setup Phase Outputs
    # ═══════════════════════════════════════════════════════════════════════════

    intent_analysis: NotRequired[str]
    world_setting: NotRequired[str]
    character_setting: NotRequired[str]
    story_outline: NotRequired[str]
    chapter_outlines: NotRequired[str]
    setup_complete: NotRequired[Annotated[bool, _last_value]]
    volume_structure: NotRequired[Annotated[dict, _last_value]]
    auto_guidance: NotRequired[Annotated[str, _last_value]]

    # ═══════════════════════════════════════════════════════════════════════════
    #  Quality Assurance
    # ═══════════════════════════════════════════════════════════════════════════

    foreshadowing_status: NotRequired[Annotated[str, _last_value]]
    pacing_status: NotRequired[Annotated[str, _last_value]]
    audit_results: NotRequired[Annotated[list, operator.add]]
    volume_status: NotRequired[Annotated[dict, _last_value]]
    quality_trend: NotRequired[Annotated[dict, _last_value]]
    guideline_complete: NotRequired[Annotated[bool, _last_value]]
    guidance_complete: NotRequired[Annotated[bool, _last_value]]

    # ═══════════════════════════════════════════════════════════════════════════
    #  Writing Phase Outputs
    # ═══════════════════════════════════════════════════════════════════════════

    # These are set by the writing_crew subgraph and consumed by parent graph
    chapter_draft: NotRequired[Annotated[str, _last_value]]
    refined_chapter: NotRequired[Annotated[str, _last_value]]
    review_result: NotRequired[Annotated[dict, _last_value]]
    quality_score: NotRequired[Annotated[float, _last_value]]
    ai_style_score: NotRequired[Annotated[float, _last_value]]
    lao_shu_chong_score: NotRequired[Annotated[float, _last_value]]
    chapter_approved: NotRequired[Annotated[bool, _last_value]]

    # ═══════════════════════════════════════════════════════════════════════════
    #  Crew & Subgraph Communication
    # ═══════════════════════════════════════════════════════════════════════════

    crew_result: NotRequired[Annotated[dict, _last_value]]

    # ═══════════════════════════════════════════════════════════════════════════
    #  Feishu Integration
    # ═══════════════════════════════════════════════════════════════════════════

    folder_tokens: NotRequired[Annotated[dict, _last_value]]

    # ═══════════════════════════════════════════════════════════════════════════
    #  Media Phase Outputs
    # ═══════════════════════════════════════════════════════════════════════════

    illustration_url: NotRequired[str]
    audio_url: NotRequired[str]

    # ═══════════════════════════════════════════════════════════════════════════
    #  Accumulated Fields
    # ═══════════════════════════════════════════════════════════════════════════

    completed_chapters: Annotated[list, _add_chapters_compressed]
    messages: Annotated[list, add_messages]
    total_usage: Annotated[dict, _add_usage]

    # ═══════════════════════════════════════════════════════════════════════════
    #  Managed Values (auto-computed, not persisted)
    # ═══════════════════════════════════════════════════════════════════════════

    chapter_progress: Annotated[dict, ChapterProgress]
    phase_status: Annotated[dict, PhaseStatus]

    # ═══════════════════════════════════════════════════════════════════════════
    #  Internal Tracking (ephemeral, cross-node communication)
    # ═══════════════════════════════════════════════════════════════════════════

    # These fields use _temp_ prefix to indicate they are NOT persisted
    # to the checkpoint. They are used for in-flight communication between nodes.
    _last_supervisor_phase: NotRequired[Annotated[str, _last_value]]
    _first_analysis_done: NotRequired[Annotated[bool, _last_value]]
    _last_monitor_report: NotRequired[Annotated[str, _last_value]]
    _last_monitor_chapter: NotRequired[Annotated[int, _last_value]]

    # ═══════════════════════════════════════════════════════════════════════════
    #  Control Fields
    # ═══════════════════════════════════════════════════════════════════════════

    current_phase: NotRequired[
        Annotated[
            Literal["setup", "writing", "media", "sync", "done"],
            _last_value,
        ]
    ]
    pending_review: NotRequired[Annotated[str, _last_value]]
    user_decision: NotRequired[Annotated[str, _last_value]]
    user_comment: NotRequired[str | None]
    modifications: NotRequired[dict | None]

    # ═══════════════════════════════════════════════════════════════════════════
    #  Human-in-the-Loop
    # ═══════════════════════════════════════════════════════════════════════════

    chapter_needs_guidance: NotRequired[Annotated[bool, _last_value]]
    human_guidance: NotRequired[Annotated[str, _last_value]]

    # ═══════════════════════════════════════════════════════════════════════════
    #  Project-Level
    # ═══════════════════════════════════════════════════════════════════════════

    quota_info: NotRequired[QuotaInfo]
    retry_count: NotRequired[int]
    word_count: NotRequired[int]

    # ═══════════════════════════════════════════════════════════════════════════
    #  Long-Term Memory
    # ═══════════════════════════════════════════════════════════════════════════

    loaded_memory: NotRequired[dict]

    # ═══════════════════════════════════════════════════════════════════════════
    #  Quota & Cost
    # ═══════════════════════════════════════════════════════════════════════════

    quota_snapshot: NotRequired[dict]
    cost_projection: NotRequired[dict]

    # ═══════════════════════════════════════════════════════════════════════════
    #  Media Phase Control
    # ═══════════════════════════════════════════════════════════════════════════

    has_media: NotRequired[Annotated[bool, _last_value]]
    media_complete: NotRequired[Annotated[bool, _last_value]]

    # ═══════════════════════════════════════════════════════════════════════════
    #  Performance Monitoring
    # ═══════════════════════════════════════════════════════════════════════════

    wall_time_data: NotRequired[Annotated[str, _last_value]]