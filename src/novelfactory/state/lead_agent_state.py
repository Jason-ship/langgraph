"""Lead Agent state — shared state for the conversational agent graph.

This state is used by the Lead Agent graph and is designed to be compatible
with the existing NovelFactoryState for seamless mode switching.
"""

from __future__ import annotations

from typing import Annotated, NotRequired, TypedDict

from langgraph.graph.message import add_messages

from novelfactory.state.reducers import (
    _add_chapters_compressed,
    _add_usage,
    _last_value,
    merge_artifacts,
    merge_delegations,
    merge_goal,
    merge_todos,
)


class SubtaskInfo(TypedDict, total=False):
    """Subtask information for SSE tracking."""
    id: str
    agent_type: str
    description: str
    status: str  # in_progress, completed, failed
    model_name: str
    usage: dict


class ArtifactInfo(TypedDict, total=False):
    """Artifact information for workspace tracking."""
    id: str
    name: str
    type: str
    content: str
    url: str


class LeadAgentState(TypedDict):
    """Lead Agent shared state — DeerFlow-compatible conversational state.

    This state is used by the Lead Agent graph for conversational interaction.
    It shares overlapping keys with NovelFactoryState (messages, total_usage, etc.)
    for seamless mode switching between batch and conversational modes.
    """

    # ═══════════════════════════════════════════════════════════════════════════
    #  Conversation (DeerFlow-compatible)
    # ═══════════════════════════════════════════════════════════════════════════

    messages: Annotated[list, add_messages]
    thread_id: str
    user_id: NotRequired[str]

    # ═══════════════════════════════════════════════════════════════════════════
    #  Agent Routing (DeerFlow-compatible)
    # ═══════════════════════════════════════════════════════════════════════════

    current_agent: NotRequired[str]
    agent_context: NotRequired[dict]

    # ═══════════════════════════════════════════════════════════════════════════
    #  Sub-task Tracking (DeerFlow-compatible)
    # ═══════════════════════════════════════════════════════════════════════════

    subtasks: NotRequired[list[SubtaskInfo]]

    # ═══════════════════════════════════════════════════════════════════════════
    #  Skills System (DeerFlow-compatible)
    # ═══════════════════════════════════════════════════════════════════════════

    active_skills: NotRequired[list[str]]
    skill_context: NotRequired[dict]

    # ═══════════════════════════════════════════════════════════════════════════
    #  Context Management (DeerFlow-compatible)
    # ═══════════════════════════════════════════════════════════════════════════

    context_summary: NotRequired[str]
    needs_compaction: NotRequired[bool]

    # ═══════════════════════════════════════════════════════════════════════════
    #  Workspace / Artifacts (DeerFlow-compatible)
    # ═══════════════════════════════════════════════════════════════════════════

    artifacts: NotRequired[Annotated[list[str], merge_artifacts]]

    # ═══════════════════════════════════════════════════════════════════════════
    #  Todo / Goal / Delegation (DeerFlow-compatible)
    # ═══════════════════════════════════════════════════════════════════════════

    todos: NotRequired[Annotated[list[dict], merge_todos]]
    goal: NotRequired[Annotated[dict, merge_goal]]
    delegations: NotRequired[Annotated[list[dict], merge_delegations]]

    # ═══════════════════════════════════════════════════════════════════════════
    #  NovelFactory Compatibility Fields
    #  These fields overlap with NovelFactoryState, allowing the Lead Agent
    #  to delegate to the batch processing pipeline when needed.
    # ═══════════════════════════════════════════════════════════════════════════

    # Project context
    project_name: NotRequired[Annotated[str, _last_value]]
    genre: NotRequired[Annotated[str, _last_value]]
    target_chapters: NotRequired[Annotated[int, _last_value]]
    current_chapter: NotRequired[Annotated[int, _last_value]]

    # Setup phase
    world_setting: NotRequired[str]
    character_setting: NotRequired[str]
    story_outline: NotRequired[str]
    chapter_outlines: NotRequired[str]
    setup_complete: NotRequired[Annotated[bool, _last_value]]

    # Writing phase
    chapter_draft: NotRequired[Annotated[str, _last_value]]
    quality_score: NotRequired[Annotated[float, _last_value]]
    crew_result: NotRequired[Annotated[dict, _last_value]]

    # Accumulated
    completed_chapters: NotRequired[Annotated[list, _add_chapters_compressed]]
    total_usage: NotRequired[Annotated[dict, _add_usage]]

    # Phase control
    # NOTE: NovelFactoryState 使用 Literal["setup","writing","media","sync","done"]，
    # 但 Lead Agent 作为对话模式需要接受更宽泛的 phase 值，
    # 因此保持 str 宽类型（与 DeerFlow 对话模式兼容）。
    current_phase: NotRequired[Annotated[str, _last_value]]
