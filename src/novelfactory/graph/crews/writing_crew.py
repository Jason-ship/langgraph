"""Writing Crew 子图 — NovelFactory 章节创作引擎。

流式输出 (StreamWriter):
    - chapter_planner: 章节写作计划
    - chapter_writer: 章节草稿
    - verdict_engine: 融合评分 (final_score + programmatic_score + verdict level)
    - chapter_refiner: 润色输出 (REFINE 路由)
    - _exit_for_chapter: 最终章节正文

拓扑 (v6.3 三路决策, verdict_router):

    START → context_builder → chapter_planner → chapter_writer → verdict_engine
                                                                      │
                                                              verdict_router
                                                                      │
            ┌─────────────────────────────────────────────────────────┼──────────────────┐
            ▼ (REWRITE, score 低)                                    ▼ (REFINE, 中等)    ▼ (PASS)
      chapter_planner                                         chapter_refiner    state_extractor
            │                                                       │                  │
            ▼                                                       ▼                  ▼
      chapter_writer                                         verdict_engine   database_writer
            │                                                                          │
            ▼                                                                          ▼
      verdict_engine                                                       __exit_for_chapter__
                                                                                  │
                                                                                 END

    路由规则 (verdict_router):
        PASS    → state_extractor_node → database_writer_node → __exit_for_chapter__ → END
        REFINE  → chapter_refiner → verdict_engine (re-review, 最多 2 次)
        REWRITE → chapter_planner → chapter_writer → verdict_engine (重新规划+重写, 最多 5 次)

    防死循环：
        - rewrite_exhausted (MAX_REWRITE_ATTEMPTS) → 强制 PASS
        - refine_exhausted (max 2) → 强制 PASS
        - v7.0 迭代宽松加分：每次重写/润色逐渐放宽评分

Usage:
    from novelfactory.graph.crews.writing_crew import build_writing_crew
"""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from novelfactory.config.constants import (
    MAX_REWRITE_ATTEMPTS,
    SUBGRAPH_RECURSION_LIMIT,
    VERDICT_REFINE_THRESHOLD,
)
from novelfactory.evaluation.coordinator import verdict_engine_node
from novelfactory.evaluation.verdict.router import verdict_router
from novelfactory.graph.crews.writing_nodes.corrector import corrector_node
from novelfactory.graph.crews.writing_nodes.critic_pre import critic_pre_assessment_node
from novelfactory.graph.crews.writing_nodes.planner import chapter_planner_node
from novelfactory.graph.crews.writing_nodes.reviewer import _chapter_refiner_node
from novelfactory.graph.crews.writing_nodes.routing import _exit_for_chapter
from novelfactory.graph.crews.writing_nodes.subgraph_integration import (
    context_builder_node_fn,
    database_writer_node_fn,
    state_extractor_node_fn,
)
from novelfactory.graph.crews.writing_nodes.writer import _chapter_writer_node
from novelfactory.state.crew_state import BaseCrewState
from novelfactory.state.novel_state import _add_usage, _last_value

# ── Local State ────────────────────────────────────────────────────────────────


class WritingCrewLocalState(BaseCrewState):
    """Local state for the Writing Crew subgraph (extends BaseCrewState).

    Design principles:
    - Persistent fields: explicitly declared, propagated to parent via _last_value
    - Temporary fields: prefixed with _temp_, cleared at subgraph exit
    - Routing fields: read by verdict_router, must be at top level
    """

    # ── Persistent Fields (propagated to parent via _last_value) ──────────────
    current_chapter: int
    target_chapters: int
    loop_count: int
    quality_score: float
    chapter_draft: str
    refine_attempts: int
    human_guidance: str
    writer_context: str
    thread_id: str
    extracted: dict
    completed_chapters: Annotated[list, _last_value]
    ai_style_score: float
    lao_shu_chong_score: float
    total_usage: Annotated[dict, _add_usage]
    review_result: dict
    verdict_result: dict
    final_score: float
    chapter_plan: dict
    story_outline: str
    character_setting: str
    world_setting: str
    chapter_outlines: str

    # ── Temporary Fields (not persisted, cleared at subgraph exit) ───────────
    # These fields are used for in-flight communication between writing nodes
    # and should NOT be propagated to the parent graph checkpoint.
    # They are prefixed with _temp_ to distinguish from persistent fields.
    _temp_toxic_points: list[str]
    _temp_shuangdian_points: list[str]
    _temp_guide_references: list[dict]
    _temp_ai_style_fix: str
    _temp_lao_shu_chong_fix: str
    _temp_debate_issues: list[str]
    _temp_debate_strengths: list[str]
    _temp_debate_suggestions: str
    _temp_is_short_text: bool
    _temp_critic_assessment: str
    _temp_critic_feedback: str
    _temp_debate_transcript: str
    _temp_ai_style_metrics_brief: str
    _temp_cross_chapter_brief: str


# ── Graph Builder ─────────────────────────────────────────────────────────────


def _rewrite_router(state: dict) -> str:
    """REWRITE 路由 — 在 rewrite 次数用尽 + 低分时先走纠偏器再重规划。

    通常 verdict_router 直接返回目标节点。本函数在其基础上增加：
    - 当 REWRITE 路径且 loop_count >= MAX_REWRITE_ATTEMPTS + 分数 < VERDICT_REFINE_THRESHOLD
      → 先走 corrector_node 注入意外事件，再走 chapter_planner
    """
    base = verdict_router(state)
    if base != "chapter_planner":
        return base

    loop_count = int(state.get("loop_count", 0))
    final_score = float(state.get("final_score", 0))
    if loop_count >= MAX_REWRITE_ATTEMPTS and final_score < VERDICT_REFINE_THRESHOLD:
        return "corrector_node"
    return "chapter_planner"


def build_writing_crew(checkpointer: Any = None) -> CompiledStateGraph:
    """Build the Writing Crew StateGraph.

    Routing pattern (all conditional edges):

      START ──→ context_builder_node ──→ chapter_writer ──→ chapter_reviewer
                                                                │
                                                       verdict_router
                                                                │
                            ├── "chapter_writer"     (score < 60, rewrite loop)
                            ├── "chapter_refiner"    (score 60-89, refine)
                            └── "__exit_for_chapter__"
                                  (双条件通过 OR loop 用尽)
                                    │
                                    ▼
                            state_extractor_node
                                    │
                                    ▼
                            database_writer_node
                                    │
                                    ▼
                            __exit_for_chapter__ ──→ END

    chapter_refiner ──→ chapter_reviewer (re-review loop)

    No node returns Command directly.  All crew-exit transitions go through
    _exit_for_chapter, which is the sole exit point.

    v5.0: chapter_reviewer 直接评分（非 quality_panel 辩论式评审）。

    Native LangGraph migration (2026-06-17):
      - Compiled WITHOUT checkpointer — parent graph handles persistence
        when this subgraph is added via add_node().
      - ContextBuilder/StateExtractor/DatabaseWriter subgraphs integrated
        as separate nodes in the graph.

    Returns:
        Compiled StateGraph.  Add as a node in the root graph:
            graph.add_node("writing_crew", build_writing_crew())
    """
    graph = StateGraph(WritingCrewLocalState)

    graph.add_node("context_builder_node", context_builder_node_fn)
    graph.add_node("chapter_planner", chapter_planner_node)
    graph.add_node("critic_pre_assessment", critic_pre_assessment_node)
    graph.add_node("corrector_node", corrector_node)
    graph.add_node("chapter_writer", _chapter_writer_node)
    graph.add_node("verdict_engine", verdict_engine_node)
    graph.add_node("chapter_refiner", _chapter_refiner_node)
    graph.add_node("state_extractor_node", state_extractor_node_fn)
    graph.add_node("database_writer_node", database_writer_node_fn)
    graph.add_node("__exit_for_chapter__", _exit_for_chapter)

    # Entry: START → context_builder_node → chapter_planner → critic_pre → chapter_writer
    graph.add_edge(START, "context_builder_node")
    graph.add_edge("context_builder_node", "chapter_planner")

    # chapter_planner → critic_pre_assessment (v7.3)
    graph.add_edge("chapter_planner", "critic_pre_assessment")

    # critic router: FAIL → 重规划, PASS/FLAG → chapter_writer
    def _critic_router(state: dict) -> str:
        assessment = state.get("critic_assessment", "PASS")
        return "chapter_planner" if assessment == "FAIL" else "chapter_writer"

    graph.add_conditional_edges(
        "critic_pre_assessment",
        _critic_router,
        {"chapter_planner": "chapter_planner", "chapter_writer": "chapter_writer"},
    )

    # chapter_writer → verdict_engine (unconditional)
    graph.add_edge("chapter_writer", "verdict_engine")

    # verdict_engine → routing via verdict_router (3 级决策)
    graph.add_conditional_edges(
        "verdict_engine",
        _rewrite_router,
        {
            "corrector_node": "corrector_node",
            "chapter_planner": "chapter_planner",
            "chapter_refiner": "chapter_refiner",
            "__exit_for_chapter__": "state_extractor_node",
        },
    )

    # corrector_node → chapter_planner（纠偏器产出事件后继续重规划）
    graph.add_edge("corrector_node", "chapter_planner")

    # chapter_refiner → verdict_engine (re-review loop)
    graph.add_edge("chapter_refiner", "verdict_engine")

    # Exit pipeline: state_extractor → database_writer → _exit_for_chapter → END
    graph.add_edge("state_extractor_node", "database_writer_node")
    graph.add_edge("database_writer_node", "__exit_for_chapter__")
    graph.add_edge("__exit_for_chapter__", END)

    # Native add_node: compile without checkpointer.
    # The parent graph's checkpointer handles all persistence.
    # The checkpointer parameter is kept for backward compat but ignored.
    compiled = graph.compile()
    compiled.recursion_limit = SUBGRAPH_RECURSION_LIMIT
    return compiled
