"""State definitions for NovelFactory LangGraph Multi-Agent Architecture.

本模块仅包含 TypedDict schema 定义（QuotaInfo / NovelFactoryState）。
自定义 reducer 和工具函数已拆分至 ``novelfactory.state.reducers``。

为保持向后兼容，reducer 仍在此模块 re-export，
现有 ``from novelfactory.state.novel_state import _last_value`` 等导入不受影响。
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
    # Reducer re-exports（向后兼容，测试和 writing_crew 直接引用）
    "_last_value",
    "_add_usage",
    "_chapter_key",
    "_add_chapters_compressed",
]


class QuotaInfo(TypedDict, total=False):
    """MiniMax API quota information — populated by QuotaClient.fetch_quota().

    Populated at project start and refreshed periodically during execution.
    Shape mirrors the dict returned by agents/infra/quota.refresh_quota().
    """

    remaining_tokens: int
    total_tokens: int
    used_tokens: int
    remaining_pct: float
    reset_time: int  # Unix timestamp (seconds)
    seconds_until_reset: int
    is_exhausted: bool
    is_critical: bool
    model_quotas: list[
        dict
    ]  # [{model_name, remaining, total, remaining_pct, used}, ...]
    raw: str  # Raw API response for debugging


class NovelFactoryState(TypedDict):
    """Global state shared across all Crews.

    This is the root state for the entire application. Each project
    is isolated by thread_id. All Crews read from and write to this state.

    Accumulated fields (lists, dicts) use Annotated with operator.add,
    add_messages, or _add_usage. Scalar fields that may be written by
    multiple nodes (e.g. thread_id, seed_idea, current_phase) use
    Annotated[T, _last_value] to avoid InvalidUpdateError on
    concurrent multi-source writes (P0-fix 2026-06-07).
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    thread_id: NotRequired[Annotated[str, _last_value]] = (
        ""  # empty-string default prevents KeyError on first run
    )
    project_context: NotRequired[ProjectContext]

    # ── User Input ───────────────────────────────────────────────────────────
    seed_idea: NotRequired[Annotated[str, _last_value]]
    genre: NotRequired[Annotated[str, _last_value]]
    project_name: NotRequired[Annotated[str, _last_value]]
    target_chapters: NotRequired[Annotated[int, _last_value]]
    current_chapter: NotRequired[Annotated[int, _last_value]]

    # ── Setup Phase Outputs ──────────────────────────────────────────────────
    intent_analysis: NotRequired[str]
    world_setting: NotRequired[str]
    character_setting: NotRequired[str]
    story_outline: NotRequired[str]
    chapter_outlines: NotRequired[str]
    setup_complete: NotRequired[Annotated[bool, _last_value]]
    # v4.1: Hierarchical outline structure (story→volume→chapter)
    # Populated by setup phase, used by post_sync_check for lazy generation
    volume_structure: NotRequired[Annotated[dict, _last_value]]

    # v4.1: Auto-guidance injected by post_sync_check node
    # Contains volume transition / quality intervention / foreshadowing
    # enforcement guidance. Propagated to writing_crew via crew_result.
    auto_guidance: NotRequired[Annotated[str, _last_value]]

    # ── Phase 2: Quality Assurance Fields (v4.1+) ──────────────────────────
    # Set by foreshadowing_check_node
    foreshadowing_status: NotRequired[Annotated[str, _last_value]]
    pacing_status: NotRequired[Annotated[str, _last_value]]
    audit_results: NotRequired[Annotated[list, operator.add]]

    # ── Phase 3: Production Control Fields (v4.1+) ─────────────────────────
    # Set by volume_check_node and quality_check_node
    volume_status: NotRequired[Annotated[dict, _last_value]]
    quality_trend: NotRequired[Annotated[dict, _last_value]]
    guideline_complete: NotRequired[Annotated[bool, _last_value]]
    guidance_complete: NotRequired[Annotated[bool, _last_value]]

    # ── Writing Phase Outputs ───────────────────────────────────────────────
    chapter_draft: NotRequired[Annotated[str, _last_value]]
    refined_chapter: NotRequired[Annotated[str, _last_value]]
    review_result: NotRequired[Annotated[dict, _last_value]]
    quality_score: NotRequired[Annotated[float, _last_value]]
    # v5.0 评分系统字段（来自 chapter_reviewer / composite_scorer）
    # 必须在父图中声明，否则子图→父图 auto-merge 会静默丢弃
    # P0-fix: 缺少这些字段导致 composite_score 在子图出口丢失，
    # verdict_router 兜底使用 quality/100=1.0 替代，所有章节直接通过
    # v6.1: composite_score 已移除，统一使用 verdict_result.programmatic_score
    ai_style_score: NotRequired[Annotated[float, _last_value]]
    lao_shu_chong_score: NotRequired[Annotated[float, _last_value]]
    chapter_approved: NotRequired[Annotated[bool, _last_value]]

    # ── Crew Result (shared with subgraphs via native add_node) ────────────
    # Carries the crew_result dict between root graph and compiled crew
    # subgraphs (writing_crew, media_crew, sync_crew). Uses _last_value
    # reducer so the most recent crew's output wins.
    # Migrated 2026-06-17: previously this field was undeclared but used
    # at runtime; now formally declared for native LangGraph subgraph
    # state sharing (add_node(compiled_subgraph) requires overlapping keys).
    crew_result: NotRequired[Annotated[dict, _last_value]]

    # ── Feishu Drive Folder Tokens ──────────────────────────────────────────
    # Created during setup, reused across sync/upload phases.
    # Shape: {"project": str, "setup": str, "chapters": str, "volume_folder_tokens": {int: str}}
    folder_tokens: NotRequired[Annotated[dict, _last_value]]

    # ── Media Phase Outputs ─────────────────────────────────────────────────
    illustration_url: NotRequired[str]
    audio_url: NotRequired[str]

    # ── Accumulated Fields ──────────────────────────────────────────────────
    # v6.0: 使用 _add_chapters_compressed 替代 operator.add，
    # 防止父图 checkpoint 在 1000+ 章时无限增长。
    # WritingCrewLocalState 仍使用 operator.add（子图内不压缩）。
    completed_chapters: Annotated[list, _add_chapters_compressed]
    messages: Annotated[list, add_messages]

    # ── Managed Values (自动计算，不持久化) ──────────────────────────────────
    chapter_progress: Annotated[dict, ChapterProgress]
    phase_status: Annotated[dict, PhaseStatus]

    # ── Internal Tracking Fields (不持久化，用于跨节点通信) ──────────────────
    # 由 main_supervisor_node 设置，用于防止 phase 转换消息重复发出
    _last_supervisor_phase: NotRequired[Annotated[str, _last_value]]
    # 由 intelligent_monitor_node 设置，跟踪首次分析是否完成
    _first_analysis_done: NotRequired[Annotated[bool, _last_value]]
    # 由 intelligent_monitor_node 设置，供调试/监控使用
    _last_monitor_report: NotRequired[Annotated[str, _last_value]]
    _last_monitor_chapter: NotRequired[Annotated[int, _last_value]]

    # ── Control Fields ──────────────────────────────────────────────────────
    current_phase: NotRequired[
        Annotated[
            Literal[
                "setup",
                "writing",
                "media",
                "sync",
                "done",
            ],
            _last_value,
        ]
    ]
    pending_review: NotRequired[Annotated[str, _last_value]]
    user_decision: NotRequired[Annotated[str, _last_value]]
    user_comment: NotRequired[str | None]
    modifications: NotRequired[dict | None]
    # NOTE: interrupt_reason and resume were removed in v6.1.
    # They were deprecated placeholders for an early interrupt design;
    # the current implementation uses wait_for_review / interrupt_recovery
    # nodes directly.

    # ── Human-in-the-Loop Guidance ─────────────────────────────────────────
    # Set by writing_crew when low score + all rewrites exhausted
    chapter_needs_guidance: NotRequired[Annotated[bool, _last_value]]
    # User-provided revision guidance from interrupt (injected into chapter_writer)
    human_guidance: NotRequired[Annotated[str, _last_value]]

    # ── Project-Level ───────────────────────────────────────────────────────
    quota_info: NotRequired[QuotaInfo]
    retry_count: NotRequired[int]
    word_count: NotRequired[int]

    # ── Long-Term Memory (BaseStore) ───────────────────────────────────────
    # Loaded from cross-session store on project start.  Shape:
    #   {
    #     "genre_templates": [...],
    #     "previous_outlines": [...],
    #     "user_preferences": {...},
    #   }
    # Written back to store on project completion.
    loaded_memory: NotRequired[dict]

    # ── Token Usage Tracking ────────────────────────────────────────────────
    # Accumulated across all chapters and phases.
    # Shape: {
    #   "prompt_tokens": int,
    #   "completion_tokens": int,
    #   "total_tokens": int,
    #   "estimated_cost_cny": float,
    #   "model_breakdown": {model_name: {"prompt_tokens": int, "completion_tokens": int, ...}},
    #   "chapter_usages": [{chapter_number: int, prompt_tokens: int, completion_tokens: int, ...}, ...],
    # }
    total_usage: Annotated[dict, _add_usage]

    # ── Quota Snapshot ──────────────────────────────────────────────────────
    # Latest billing API snapshot. Shape: see tools/quota_api.py.
    quota_snapshot: NotRequired[dict]

    # ── Cost Projection ─────────────────────────────────────────────────────
    # Per-chapter average cost + projected remaining cost.
    # Shape: {
    #   "chapters_completed": int,
    #   "avg_cost_per_chapter_cny": float,
    #   "projected_remaining_cost_cny": float,
    #   "chapters_remaining": int,
    # }
    cost_projection: NotRequired[dict]

    # ── Media Phase ──────────────────────────────────────────────────────
    # Whether the project requires media generation (illustration/audio).
    # Set at project start, checked by supervisor to route to media_crew.
    has_media: NotRequired[Annotated[bool, _last_value]]
    # Set by media_crew upon completion (True = success, False = all retries exhausted).
    media_complete: NotRequired[Annotated[bool, _last_value]]

    # wall_time_data: serialized WallTimeTracker JSON for performance monitoring.
    wall_time_data: NotRequired[Annotated[str, _last_value]]
