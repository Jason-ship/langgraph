"""Lightweight Setup Supervisor — no Crew Supervisor overhead.

Directly calls each agent sequentially without LLM-driven routing.
LLM call budget:
    WorldBuilder       → 1 call  (via create_react_agent with RAG tools)
    CharacterDesigner  → 1 call
    OutlineWriter      → 1 call
    setup_quality_gate → 1 call (M3-5: LLM reviewer, replaces char-count scoring)
    Total: 4 LLM calls  (vs 11 calls with Crew Supervisor)

M3-5: Quality gate uses LLM reviewer (get_reviewer_llm) instead of char-count
heuristics. Scoring dimensions: 世界观完整度(30) + 角色立体度(25) +
大纲结构(25) + 设定自洽性(20). Threshold ≥70 to pass.

Streaming: writes partial output to a temp file after each agent finishes.
"""

from __future__ import annotations

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from novelfactory.agents.infra import (
    get_logger,
    llm_call_with_retry,
    validate_json_output,
)
from novelfactory.config.constants import SUBGRAPH_RECURSION_LIMIT
from novelfactory.config.llm import get_reviewer_llm
from novelfactory.state.crew_state import BaseCrewState

_logger = get_logger("novelfactory.graph.lightweight_setup")

# M3-5: Setup quality gate threshold
_SETUP_QUALITY_THRESHOLD = 70.0
_CHAPTER_RANGE_SIZE_START = 1  # chapter_range 列表最小元素数（访问索引 0 用）
_CHAPTER_RANGE_SIZE_END = 2  # chapter_range 列表最小元素数（访问索引 1 用）


def _retry_invoke(agent: Runnable, input_dict: dict, step_name: str) -> dict:
    """Production-grade agent.invoke with timeout + exponential-backoff retry."""
    result = llm_call_with_retry(
        agent.invoke,
        input_dict,
        step_name=f"lightweight_setup.{step_name}",
        fallback={"messages": [], "crew_result": {}},
    )
    return result if result is not None else {"messages": [], "crew_result": {}}


async def _llm_quality_gate(
    world_setting: str,
    character_setting: str,
    story_outline: str,
    chapter_outlines: str,
) -> tuple[float, str]:
    """M3-5: Score setup outputs using LLM reviewer (v5.4: 异步化 + 重试保护).

    Returns (quality_score, review_comments).
    Uses get_reviewer_llm (M2.5) for efficient scoring — output is only ~500 tokens.
    v5.4: 改用 async_llm_call_with_retry 包装，获得超时+重试+熔断保护。
    """
    from novelfactory.agents.infra.async_retry import async_llm_call_with_retry

    llm = get_reviewer_llm()
    prompt = f"""请审核以下 Setup 阶段产出，给出量化评分。

【世界观设定】（字数：{len(world_setting)}）
{world_setting[:3000]}

【角色设定】（字数：{len(character_setting)}）
{character_setting[:2000]}

【故事主线】（字数：{len(story_outline)}）
{story_outline[:2000]}

【章节大纲】（字数：{len(chapter_outlines)}）
{chapter_outlines[:2000]}

## 评分维度（总分100）
| 维度 | 满分 | 评分锚点 |
|------|------|----------|
| 世界观完整度 | 30分 | 地理/力量体系/社会结构/历史均有详细描述 |
| 角色立体度 | 25分 | 人物有明确性格、动机、成长弧线 |
| 大纲结构 | 25分 | 起承转合清晰，有明确冲突和悬念 |
| 设定自洽性 | 20分 | 世界观与角色行为完全自洽 |

## 字数门槛
- 世界观设定 ≥3000字（不足按比例扣分）
- 角色设定 ≥1500字（不足按比例扣分）
- 大纲 ≥2000字（不足按比例扣分）

请先在 <thinking> 标签内逐维分析，然后输出JSON。
你必须输出有效的JSON，不要输出其他内容。
JSON格式：{{"quality_score": <总分>, "review_comments": "<评分明细+改进建议>"}}
"""

    # v5.4: 异步 LLM 调用 + 重试保护
    async def _invoke():
        response = await llm.ainvoke([("user", prompt)])
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )
        # Record usage
        if hasattr(response, "response_metadata"):
            meta = response.response_metadata or {}
            usage = meta.get("usage") or meta.get("token_usage", {})
            if isinstance(usage, dict):
                pt = int(usage.get("prompt_tokens", 0) or 0)
                ct = int(usage.get("completion_tokens", 0) or 0)
                if pt or ct:
                    from novelfactory.agents.infra.usage import _record_usage

                    _record_usage("setup_quality_gate", pt, ct)
                    _logger.info(
                        "setup_quality_gate usage: prompt=%d completion=%d total=%d",
                        pt,
                        ct,
                        pt + ct,
                    )
        return {"messages": [response], "_text": response_text}

    result = await async_llm_call_with_retry(
        _invoke,
        step_name="setup_quality_gate",
        timeout_seconds=120,
        fallback={"messages": [], "_text": ""},
    )
    text = result.get("_text", "") if isinstance(result, dict) else ""

    parsed, err = validate_json_output(
        text,
        required_keys=["quality_score", "review_comments"],
        fail_closed=False,
    )
    if parsed:
        score = float(parsed.get("quality_score", 50.0))
        comments = str(parsed.get("review_comments", ""))
        return max(0.0, min(100.0, score)), comments
    return 50.0, f"评分解析失败：{err or text[:200]}"


# ── Volume Structure Persistence ──────────────────────────────────────────────


def _persist_volume_structure_to_db(
    project_name: str,
    volume_structure: dict,
    first_3_volumes_detail: list[dict],
) -> None:
    """将卷级大纲结构和前3卷的章节大纲持久化到数据库。

    使用 OutlineManager 将卷和章大纲写入 novel_volumes 和 novel_chapter_outlines 表。
    如果数据库不可用，静默跳过（不影响 Setup 流程）。

    Args:
        project_name: 项目名称。
        volume_structure: 卷级大纲结构（含 volumes 列表）。
        first_3_volumes_detail: 前3卷的详细章大纲列表，每个元素含 chapter_outlines 字段。
    """
    try:
        from novelfactory.config.database import DatabaseManager
        from novelfactory.pipeline.scale_manager import ChapterOutline, OutlineManager

        with DatabaseManager.get_instance().get_connection() as conn:
            manager = OutlineManager(conn)

            volumes = volume_structure.get("volumes", [])
            # 构建前3卷的章节大纲映射 {volume_number: [chapter_dicts]}
            detail_map: dict[int, list[dict]] = {
                v["volume_number"]: v.get("chapter_outlines", [])
                for v in first_3_volumes_detail
            }

            for vol in volumes:
                vol_num = vol.get("volume_number", 0)
                ch_range = vol.get("chapter_range", [0, 0])
                start_ch = (
                    ch_range[0]
                    if isinstance(ch_range, list) and len(ch_range) >= 1
                    else 0
                )
                end_ch = (
                    ch_range[1]
                    if isinstance(ch_range, list)
                    and len(ch_range) >= _CHAPTER_RANGE_SIZE_END
                    else 0
                )

                manager.create_volume(
                    project=project_name,
                    volume_number=vol_num,
                    title=vol.get("title", f"第{vol_num}卷"),
                    theme=vol.get("theme", ""),
                    summary=vol.get("summary", ""),
                    start_chapter=start_ch,
                    end_chapter=end_ch,
                )

                # 前3卷保存章节大纲
                if vol_num in detail_map:
                    for ch in detail_map[vol_num]:
                        outline = ChapterOutline(
                            chapter_number=ch.get("chapter_number", 0),
                            volume_number=vol_num,
                            title=ch.get("title", ""),
                            goal=ch.get("core_events", ""),
                            key_beats=[],
                            pov_character="",
                            characters_involved=[],
                            foreshadowing_plant=[],
                            foreshadowing_resolve=[],
                            word_count_target=3000,
                            status="pending",
                        )
                        manager.save_chapter_outline(project_name, outline)

            _logger.info(
                "卷级大纲已持久化到数据库（%d 卷，前3卷含章大纲）",
                len(volumes),
            )
    except Exception as e:
        _logger.warning("卷级大纲持久化失败（不影响 Setup 流程）: %s", e)


# ── Setup Crew State ──────────────────────────────────────────────────────────


class SetupCrewState(BaseCrewState):
    """Local state for the Setup Crew subgraph (extends BaseCrewState).

    Inherits ``messages``, ``crew_result``, ``crew_error`` from BaseCrewState.
    Mirrors the fields required by run_lightweight_setup_supervisor() so that
    the parent graph can pass inputs via state and read outputs after completion.

    v5.4: 新增 _streaming_path 字段用于多节点流式输出。
    """

    project_name: str
    genre: str
    seed_idea: str
    target_chapters: int
    thread_id: str
    enable_streaming: bool
    world_setting: str
    character_setting: str
    story_outline: str
    chapter_outlines: str
    volume_structure: dict
    first_3_volumes_detail: list[dict]
    quality_score: float
    quality_comments: str
    review_text: str
    folder_tokens: dict
    setup_complete: bool
    setup_usage: dict
    total_usage: dict
    current_chapter: int
    current_phase: str
    _streaming_path: str  # v5.4: 流式输出临时文件路径


# ═══════════════════════════════════════════════════════════════════════════
#  v5.4 Multi-node Setup Crew (preferred — multi-node pipeline)
# ═══════════════════════════════════════════════════════════════════════════


def build_setup_crew_multi_node() -> CompiledStateGraph:
    """Build the Setup Crew as a compiled subgraph with 9 independent nodes (v5.4).

    Pipeline:
      init_setup → world_builder → character_designer → outline_writer
      → volume_detail_writer → quality_gate → feishu_setup → db_persist
      → setup_finalize

    Each node reads/writes fields in SetupCrewState. Intermediate results
    flow between nodes through state fields (world_setting, character_setting,
    story_outline, etc.). Checkpointing happens at each node boundary —
    recovery can resume from any step.

    The parent graph wires this with::

        graph.add_node("setup_crew", build_setup_crew_multi_node())

    State fields with matching names (world_setting, character_setting, etc.)
    automatically flow between the subgraph and the parent NovelFactoryState.
    """
    from novelfactory.graph.nodes.setup_nodes import (
        character_designer_node,
        db_persist_node,
        feishu_setup_node,
        init_setup_node,
        outline_writer_node,
        quality_gate_node,
        setup_finalize_node,
        volume_detail_writer_node,
        world_builder_node,
    )

    builder = StateGraph(SetupCrewState)

    builder.add_node("init_setup", init_setup_node)
    builder.add_node("world_builder", world_builder_node)
    builder.add_node("character_designer", character_designer_node)
    builder.add_node("outline_writer", outline_writer_node)
    builder.add_node("volume_detail_writer", volume_detail_writer_node)
    builder.add_node("quality_gate", quality_gate_node)
    builder.add_node("feishu_setup", feishu_setup_node)
    builder.add_node("db_persist", db_persist_node)
    builder.add_node("setup_finalize", setup_finalize_node)

    # Sequential pipeline
    builder.add_edge(START, "init_setup")
    builder.add_edge("init_setup", "world_builder")
    builder.add_edge("world_builder", "character_designer")
    builder.add_edge("character_designer", "outline_writer")
    builder.add_edge("outline_writer", "volume_detail_writer")
    builder.add_edge("volume_detail_writer", "quality_gate")
    builder.add_edge("quality_gate", "feishu_setup")
    builder.add_edge("feishu_setup", "db_persist")
    builder.add_edge("db_persist", "setup_finalize")
    builder.add_edge("setup_finalize", END)

    compiled = builder.compile()
    compiled.recursion_limit = SUBGRAPH_RECURSION_LIMIT
    return compiled


# ═══════════════════════════════════════════════════════════════════════════
#  build_setup_crew() — v5.4: 切换到多节点管线
# ═══════════════════════════════════════════════════════════════════════════


def build_setup_crew() -> CompiledStateGraph:
    """Build the Setup Crew as a compiled subgraph.

    v5.4: Single-node pipeline (deprecated).
    v5.4: Switched to multi-node pipeline — 9 independent nodes with
          intermediate checkpointing. See build_setup_crew_multi_node().

    The parent graph wires this with::

        graph.add_node("setup_crew", build_setup_crew())
    """
    return build_setup_crew_multi_node()


# ── Text extraction helpers ────────────────────────────────────────────────────


def _split_outline(full_output: str) -> tuple[str, str]:
    """Split agent output into story_outline and chapter_outlines.

    Priority: explicit "## 章节大纲" heading → numbered chapter boundaries
    → fallback: split at ~70% (story overview first, chapter details last).
    Never returns (full, full) — that would double-count tokens and confuse
    the quality gate.
    """
    for marker in [
        "## 章节大纲",
        "## 章节划分",
        "## Chapter Outline",
        "### 章节大纲",
        "### 章节划分",
    ]:
        if marker in full_output:
            parts = full_output.split(marker, 1)
            return parts[0].strip(), (marker + "\n" + parts[1].strip()).strip()

    # Secondary marker: 第N章 / Chapter 1.  boundaries
    import re as _re

    if _re.search(r"(?:第\s*\d+\s*[章章节]|Chapter\s*\d)", full_output):
        match = _re.search(r"(?:第\s*\d+\s*[章章节]|Chapter\s*\d)", full_output)
        if match:
            idx = match.start()
            return full_output[:idx].strip(), full_output[idx:].strip()

    # Final fallback: 70/30 split (assume story overview first, chapter detail last).
    fallback_idx = int(len(full_output) * 0.7)
    # Round to nearest line break to avoid mid-sentence cuts.
    nl_pos = full_output.find("\n", fallback_idx)
    if nl_pos == -1 or nl_pos >= len(full_output) - 50:
        nl_pos = fallback_idx
    return full_output[:nl_pos].strip(), full_output[nl_pos:].strip()
