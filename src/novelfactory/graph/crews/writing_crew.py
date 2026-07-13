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

    Inherits ``messages``, ``crew_result``, ``crew_error`` from BaseCrewState.

    Routing is driven by verdict_router reading verdict_result.level.
    programmatic_score = (lao_shu_chong_score / 100) * (1 - ai_style_score)

    NOTE: chapter_draft is a top-level field (not nested in crew_result) because
    LangGraph's dict reducer requires top-level fields to be updated atomically.
    Nested dict merges (crew_result) have reducer ambiguity with RunnableLambda nodes.
    """

    current_chapter: int
    target_chapters: int
    loop_count: int  # consecutive score<60 attempts on current chapter
    quality_score: float  # 四维评分，来自 verdict_engine；被 verdict_router 读取
    chapter_draft: str  # top-level for reliable reducer updates (not in crew result)
    refine_attempts: (
        int  # refine attempts for current chapter; 80-89: max 1, 60-79: max 2
    )
    human_guidance: str  # user-provided revision guidance (set via interrupt)
    writer_context: (
        str  # Context from ContextBuilder subgraph (set by context_builder_node)
    )
    # v6.1 P3-10: 允许 Send 并行分发时传递完整上下文
    thread_id: str  # 父图 thread_id，用于并行写作时的上下文隔离
    extracted: dict  # Extracted state from StateExtractor subgraph (set by state_extractor_node)
    completed_chapters: Annotated[
        list, _last_value
    ]  # v6.1-fix: use _last_value to prevent exponential growth with operator.add

    # === v5.0 评分系统字段 ===
    ai_style_score: float  # AI味指数，0-1，越低越好，≤0.3合格
    lao_shu_chong_score: float  # 老书虫视角评分，0-100，越高越好，≥70合格
    # v6.1: composite_score 已移除，统一使用 verdict_result.programmatic_score
    toxic_points: list[str]  # 检测到的毒点类型列表
    shuangdian_points: list[str]  # 检测到的爽点类型列表
    guide_references: list[dict]  # 引用的写作指南
    ai_style_fix: str  # AI味修改建议
    lao_shu_chong_fix: str  # 老书虫视角修改建议
    # v5.11: 辩论定性评审字段（editor↔reader debate，传递给 refiner/writer）
    debate_issues: list[str]  # 编辑+读者合并问题列表
    debate_strengths: list[str]  # 编辑+读者合并亮点列表
    debate_suggestions: str  # 编辑+读者合并改进建议
    # v5.1.1-fix: 声明 total_usage 使 _exit_for_chapter 的顶层返回值
    # 能通过子图→父图合并被 NovelFactoryState._add_usage reducer 处理。
    # 之前未声明导致所有写作章节的 token 数据被 LangGraph 静默丢弃。
    total_usage: Annotated[dict, _add_usage]
    # v6.0: 短文本标记 — 程序化子Agent都无法分析时绕过程序化评分
    is_short_text: bool
    # v6.0.1 P0-fix: 以下字段必须声明，否则 Send 并行分发时
    # LangGraph 会将根图→子图的这些字段静默丢弃
    story_outline: str  # 主线大纲+金手指体系
    character_setting: str  # 角色性格/说话方式设计
    world_setting: str  # 世界观设定
    chapter_outlines: str  # 章节大纲
    review_result: dict  # 上一轮评审结果（reviewer→writer 反馈链）
    # v6.3: 统一评审决议（evaluation 模块唯一权威产出）
    verdict_result: dict  # VerdictResult.model_dump()
    final_score: float  # 融合后最终评分（从 verdict_result 同步）
    # v6.3: 章节写作计划（chapter_planner 节点产出）
    chapter_plan: dict  # ChapterPlan.model_dump()
    # v7.3: Critic 前置大纲评估
    critic_assessment: str  # PASS / FLAG / FAIL
    critic_feedback: str  # 评估反馈文本
    # v6.3: 辩论记录和程序化指标（翻修 prompt 消费）
    debate_transcript: str  # 完整辩论记录
    ai_style_metrics_brief: str  # AI味 8 维指标摘要
    cross_chapter_brief: str  # 跨章一致性摘要


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
