"""
ContextBuilder 子图 — 写前上下文构建
=====================================
将 NovelStateTracker.before_chapter() 和 ScaleManager.build_writer_context()
拆解为 LangGraph 子图节点，所有数据走 state 通道。

节点:
  load_state     — 从 PG/Milvus/Neo4j 加载状态
  build_context  — 构建分层上下文（大纲+滑动窗口+弧线+审计+伏笔+节奏+成本+质量+卷）
  aggregate      — 合并为 writer_context 字符串

用法:
    from novelfactory.graph.subgraphs.context_builder import build_context_builder
    context_builder = build_context_builder()
    # 作为 writing_crew 的子图节点使用
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


# ── State ──────────────────────────────────────────────────────────────────────


class ContextBuilderState(TypedDict):
    """ContextBuilder 子图的状态。

    输入: project_name, chapter_number (从父图 crew_result 传入)
    输出: writer_context (注入 writer prompt 的完整上下文字符串)
    """

    project_name: str
    chapter_number: int

    # ── 从外部加载的原始数据 ──
    character_states: dict  # PG: 最新角色状态
    open_threads: list[dict]  # PG: 开放剧情线索
    similar_chapters: list[dict]  # Milvus: 相似章节
    character_network: list[str]  # Neo4j: 角色列表

    # ── v7.3: 需求驱动的选择性加载
    # 受 RaPID (中科大+华为 2025) 的"属性约束搜索"启发。
    # 差异：RaPID 做搜索查询分解，本实现做数据源选择，两者语义层级不同。
    chapter_needs: (
        dict  # {needs_character_network: bool, needs_similar_chapters: bool, ...}
    )

    # ── Phase1: 分层大纲 + 滑动上下文 + 弧线 ──
    chapter_outline: dict  # 当前章大纲
    volume_info: dict  # 当前卷信息
    sliding_window: str  # 滑动上下文窗口
    arc_status: str  # 角色弧线状态

    # ── Phase2: 审计 + 伏笔 + 节奏 ──
    audit_status: str  # 一致性审计状态
    foreshadowing_status: str  # 伏笔状态
    pacing_status: str  # 节奏状态

    # ── Phase3: 断点 + 成本 + 质量 + 卷 ──
    recovery_status: str  # 断点续写状态
    cost_status: str  # 成本状态
    quality_status: str  # 质量趋势
    volume_status: str  # 卷管理状态

    # ── 最终输出 ──
    writer_context: str  # 合并后的完整上下文
    error: str  # 错误信息


# ── Node: load_state ───────────────────────────────────────────────────────────


def _analyze_chapter_needs(
    project: str,
    chapter: int,
) -> dict:
    """分析当前章节的需求，决定需要加载哪些数据源。

    v7.3: 需求驱动的选择性加载 — 受 RaPID (中科大+华为 2025) 启发，但非复现：
      - RaPID Section 3.3 的"属性约束搜索"将大纲拆解为原子概念用于搜索查询。
      - 本实现将大纲拆解为数据源需求用于选择性加载（语义层级不同）。
    先分析大纲→再查多源，避免全量加载。

    Returns:
        dict with keys: needs_character_states, needs_open_threads,
        needs_similar_chapters, needs_character_network,
        needs_foreshadowing, needs_pacing, needs_quality
    """
    from novelfactory.agents.infra.retry import llm_call_with_retry

    # 加载大纲做分析
    outline_text = ""
    try:
        from novelfactory.store.tracker import NovelStateTracker

        tracker = NovelStateTracker(project)
        scale = tracker.scale
        if scale:
            outline = scale.outline.get_chapter_outline(project, chapter)
            if outline:
                outline_text = (
                    f"{outline.title} {outline.goal} {' '.join(outline.key_beats)}"
                )
        tracker.close()
    except Exception:
        # 分析失败时全量加载（保守降级）
        return {
            "needs_character_states": True,
            "needs_open_threads": True,
            "needs_similar_chapters": True,
            "needs_character_network": True,
            "needs_foreshadowing": True,
            "needs_pacing": True,
            "needs_quality": True,
        }

    if not outline_text:
        return {
            "needs_character_states": True,
            "needs_open_threads": True,
            "needs_similar_chapters": True,
            "needs_character_network": True,
            "needs_foreshadowing": False,
            "needs_pacing": False,
            "needs_quality": False,
        }

    prompt = (
        f"分析第{chapter}章的大纲，判断写作时需要哪些上下文数据。\n\n"
        f"大纲: {outline_text[:2000]}\n\n"
        f"输出 JSON 格式（不要解释）：\n"
        f"{{\n"
        f'  "needs_character_states": true/false,  // 需要角色最新状态？\n'
        f'  "needs_open_threads": true/false,       // 需要开放剧情线索？\n'
        f'  "needs_similar_chapters": true/false,   // 需要检索相似章节？\n'
        f'  "needs_character_network": true/false,  // 需要角色关系网络？\n'
        f'  "needs_foreshadowing": true/false,      // 需要检查伏笔？\n'
        f'  "needs_pacing": true/false,             // 需要节奏分析？\n'
        f'  "needs_quality": true/false             // 需要质量趋势？\n'
        f"}}"
    )

    try:
        from novelfactory.config.llm import get_reviewer_llm

        llm = get_reviewer_llm()
        response = llm_call_with_retry(llm, prompt, step_name="analyze_chapter_needs")
        raw = response.content if hasattr(response, "content") else str(response)
        import json

        needs = json.loads(raw)
        if isinstance(needs, dict):
            return needs
    except Exception:
        pass

    # 降级：全量加载
    return {
        "needs_character_states": True,
        "needs_open_threads": True,
        "needs_similar_chapters": True,
        "needs_character_network": True,
        "needs_foreshadowing": True,
        "needs_pacing": True,
        "needs_quality": True,
    }


def _load_state_node(state: ContextBuilderState) -> dict:
    """从三库加载所有状态数据。

    这是 ContextBuilder 的入口节点，替代 NovelStateTracker.before_chapter() 的
    数据库查询部分。所有查询结果写入 state，由后续节点使用。

    v7.3: 先分析章节需求（属性驱动检索），按需加载，而非全量查一圈。
    """
    project: str = state.get("project_name", "")
    chapter: int = state.get("chapter_number", 1)
    result: dict[str, Any] = {}

    # v7.3: 先分析需求
    needs = _analyze_chapter_needs(project, chapter)
    result["chapter_needs"] = needs
    logger.info(
        "[ContextBuilder] 第%d章需求分析: character_states=%s similar=%s network=%s "
        "foreshadow=%s pacing=%s quality=%s",
        chapter,
        needs.get("needs_character_states"),
        needs.get("needs_similar_chapters"),
        needs.get("needs_character_network"),
        needs.get("needs_foreshadowing"),
        needs.get("needs_pacing"),
        needs.get("needs_quality"),
    )

    tracker = None
    try:
        from novelfactory.store.tracker import NovelStateTracker

        tracker = NovelStateTracker(project)

        # PG: 角色状态 — 只在需要时加载
        if needs.get("needs_character_states", True):
            try:
                states = tracker.pg.get_latest_character_states(project)
                result["character_states"] = states
            except Exception as e:
                logger.warning("load character_states failed: %s", e)
                result["character_states"] = {}
        else:
            result["character_states"] = {}

        # PG: 开放线索 — 只在需要时加载
        if needs.get("needs_open_threads", True):
            try:
                threads = tracker.pg.get_open_threads(project)
                result["open_threads"] = threads or []
            except Exception as e:
                logger.warning("load open_threads failed: %s", e)
                result["open_threads"] = []
        else:
            result["open_threads"] = []

        # Milvus: 相似章节 — 只在需要时加载
        if needs.get("needs_similar_chapters", True):
            try:
                if tracker.milvus.is_connected():
                    from novelfactory.store.tracker import EmbeddingService

                    query_text = _build_similar_chapter_query(
                        project,
                        chapter,
                        result.get("open_threads", []),
                        result.get("character_states", {}),
                    )
                    emb_service = EmbeddingService()
                    query_vec = emb_service.embed(query_text)
                    similar = (
                        tracker.milvus.search_similar(
                            query_vec, top_k=5, project=project
                        )
                        if any(query_vec)
                        else []
                    )
                    # v7.3: 情感一致性过滤 — 过滤情感弧线不一致的检索结果
                    if similar:
                        from novelfactory.evaluation.programmatic import (
                            SentimentConsistencyFilter,
                        )

                        sentiment_filter = SentimentConsistencyFilter()
                        query_sentiment = sentiment_filter.estimate_sentiment(
                            query_text
                        )
                        filtered = sentiment_filter.filter(
                            similar, query_sentiment=query_sentiment
                        )
                        if filtered:
                            similar = filtered
                    result["similar_chapters"] = similar or []
                else:
                    result["similar_chapters"] = []
            except Exception as e:
                logger.warning("load similar_chapters failed: %s", e)
                result["similar_chapters"] = []
        else:
            result["similar_chapters"] = []

        # Neo4j: 角色网络 — 只在需要时加载
        if needs.get("needs_character_network", True):
            try:
                if tracker.neo4j.is_connected():
                    chars = tracker.neo4j.get_all_characters()
                    result["character_network"] = chars or []
                else:
                    result["character_network"] = []
            except Exception as e:
                logger.warning("load character_network failed: %s", e)
                result["character_network"] = []
        else:
            result["character_network"] = []

    except Exception as e:
        logger.error("ContextBuilder load_state failed: %s", e)
        result["error"] = str(e)
    finally:
        if tracker is not None:
            tracker.close()

    return result


# ── Node: build_context ────────────────────────────────────────────────────────


def _build_context_node(state: ContextBuilderState) -> dict:
    """构建所有分层上下文。

    替代 ScaleManager.build_writer_context() 中的各子调用。
    每个上下文独立构建，写入 state 的对应字段。
    """
    project: str = state.get("project_name", "")
    chapter: int = state.get("chapter_number", 1)
    result: dict[str, Any] = {}

    tracker = None
    try:
        from novelfactory.store.tracker import NovelStateTracker

        tracker = NovelStateTracker(project)
        scale: Any = tracker.scale

        if scale:
            # Phase1: 大纲
            try:
                outline = scale.outline.get_chapter_outline(project, chapter)
                if outline:
                    result["chapter_outline"] = {
                        "title": outline.title,
                        "goal": outline.goal,
                        "key_beats": outline.key_beats,
                        "pov_character": outline.pov_character,
                        "characters_involved": outline.characters_involved,
                        "foreshadowing_plant": outline.foreshadowing_plant,
                        "foreshadowing_resolve": outline.foreshadowing_resolve,
                    }
            except Exception as e:
                logger.warning("build chapter_outline failed: %s", e)

            # Phase1: 滑动窗口
            try:
                ctx = scale.context.build_context(project, chapter, scale.outline)
                result["sliding_window"] = ctx or ""
            except Exception as e:
                logger.warning("build sliding_window failed: %s", e)

            # Phase1: 弧线
            try:
                arc = scale.arcs.build_arc_prompt(project)
                result["arc_status"] = arc or ""
            except Exception as e:
                logger.warning("build arc_status failed: %s", e)

            # Phase1: 卷信息
            try:
                vol = scale.outline.get_current_volume(project, chapter)
                if vol:
                    result["volume_info"] = {
                        "number": vol.volume_number,
                        "title": vol.title,
                        "theme": vol.theme,
                        "summary": vol.summary,
                    }
            except Exception as e:
                logger.warning("build volume_info failed: %s", e)

            # Phase2
            try:
                if scale.phase2:
                    p2_ctx = scale.phase2.build_writer_context(chapter)
                    # 解析各子部分
                    result["audit_status"] = _extract_section(p2_ctx, "一致性审计")
                    result["foreshadowing_status"] = _extract_section(p2_ctx, "伏笔")
                    result["pacing_status"] = _extract_section(p2_ctx, "节奏")
            except Exception as e:
                logger.warning("build phase2 context failed: %s", e)

            # Phase3
            try:
                if scale.phase3:
                    p3_ctx = scale.phase3.build_writer_context(chapter)
                    result["recovery_status"] = _extract_section(p3_ctx, "续写")
                    result["cost_status"] = _extract_section(p3_ctx, "成本")
                    result["quality_status"] = _extract_section(p3_ctx, "质量")
                    result["volume_status"] = _extract_section(p3_ctx, "当前卷")
            except Exception as e:
                logger.warning("build phase3 context failed: %s", e)

    except Exception as e:
        logger.error("ContextBuilder build_context failed: %s", e)
        result["error"] = str(e)
    finally:
        if tracker is not None:
            tracker.close()

    return result


def _extract_section(text: str, keyword: str) -> str:
    """从上下文字符串中提取包含关键词的段落。"""
    if not text:
        return ""
    lines = text.split("\n")
    result_lines = []
    capturing = False
    for line in lines:
        if keyword in line:
            capturing = True
        if capturing:
            if line.strip().startswith("【") and keyword not in line:
                break
            result_lines.append(line)
    return "\n".join(result_lines) if result_lines else ""


def _build_similar_chapter_query(
    project: str,
    chapter: int,
    open_threads: list[dict],
    character_states: dict,
) -> str:
    """为 Milvus 相似章节检索构建有语义信号的 query。

    旧实现用「第N章 上下文检索」作为 query，本不含剧情/角色信息，向量
    检索结果近似随机。改为拼接项目名、主要出场角色、开放剧情线索描述，
    让 embedding 真正反映当前章写作语境，从而召回内容相关的历史章节。
    """
    parts: list[str] = [project, f"第{chapter}章"]

    # 主要角色（最多 5 个，取 PG 中已有状态的角色）
    names = [n for n in character_states.keys() if n][:5]
    if names:
        parts.append("角色：" + "、".join(names))

    # 开放剧情线索（最多 3 条，截断到 80 字）
    for t in (open_threads or [])[:3]:
        desc = str(t.get("description", "")).strip()
        if desc:
            parts.append(desc[:80])

    return "\n".join(parts)


def _build_cross_chapter_tracking(
    character_states: dict,
    open_threads: list[dict],
) -> str:
    """Build cross-chapter character and plot thread tracking context.

    Uses PG character states (already loaded by _load_state_node) to
    derive cross-chapter character movement, active/inactive status,
    and unresolved plot thread continuity — replacing the standalone
    ChapterStateTracker memory object with stateless PG-backed logic.
    """
    parts = []

    # Character states with cross-chapter tracking
    if character_states:
        lines = ["【跨章角色状态追踪】"]
        # Sort: active characters first (status != 未出场), then by name
        sorted_chars = sorted(
            character_states.items(),
            key=lambda x: (
                0 if x[1].get("status", "未出场") != "未出场" else 1,
                x[0],
            ),
        )
        active_count = 0
        for name, info in sorted_chars:
            if active_count >= 25:
                break
            loc = info.get("location", "?")
            status = info.get("status", "健在")
            mood = info.get("mood", "")
            power = info.get("power_level", "")
            items = info.get("items", [])

            line = f"  - {name}"
            if loc and loc != "未知":
                line += f" | {loc}"
            if status not in ("健在",):
                line += f" | [{status}]"
            if power and power != "未知":
                line += f" | {power}"
            if mood:
                line += f" | 心境：{mood}"
            if items:
                line += f" | 持有：{'、'.join(items[:3])}"
            lines.append(line)
            active_count += 1

        parts.append("\n".join(lines))

    # Unresolved threads with cross-chapter continuity
    if open_threads:
        lines = ["【跨章待处理线索】"]
        for t in open_threads[:15]:
            name = t.get("thread_name", "?")
            desc = str(t.get("description", ""))[:120]
            created = t.get("created_chapter", "?")
            lines.append(f"  [{name}] (第{created}章起) {desc}")
        if len(open_threads) > 8:
            lines.append(f"  ...还有 {len(open_threads) - 8} 条线索")
        parts.append("\n".join(lines))

    return "\n".join(parts) if parts else ""


# ── Node: aggregate ────────────────────────────────────────────────────────────


def _aggregate_node(state: ContextBuilderState) -> dict:
    """合并所有上下文为 writer_context 字符串。

    替代 ScaleManager.build_writer_context() 的最终合并逻辑。
    输出注入 writer prompt 的完整上下文。
    """
    parts = []

    # 1. 分层大纲
    outline: dict[str, Any] = state.get("chapter_outline", {}) or {}
    if outline:
        ch: int = state.get("chapter_number", 1)
        parts.append(f"【本章大纲】第{ch}章《{outline.get('title', '')}》")
        if outline.get("goal"):
            parts.append(f"  目标：{outline['goal']}")
        beats = outline.get("key_beats", [])
        if beats:
            parts.append(f"  节拍：{' → '.join(beats)}")
        if outline.get("pov_character"):
            parts.append(f"  主视角：{outline['pov_character']}")
        chars = outline.get("characters_involved", [])
        if chars:
            parts.append(f"  出场角色：{'、'.join(chars)}")
        plant = outline.get("foreshadowing_plant", [])
        if plant:
            parts.append(f"  需埋伏笔：{'、'.join(plant)}")
        resolve = outline.get("foreshadowing_resolve", [])
        if resolve:
            parts.append(f"  需回收伏笔：{'、'.join(resolve)}")

    # 2. 跨章角色状态追踪 + 线索连续性（增强版，替代 ChapterStateTracker 内存对象）
    char_states: dict[str, Any] = state.get("character_states", {}) or {}
    open_threads: list[dict[str, Any]] = state.get("open_threads", []) or []
    if char_states or open_threads:
        tracking_section = _build_cross_chapter_tracking(char_states, open_threads)
        if tracking_section:
            parts.append(tracking_section)

    # 3. 相似章节
    similar: list[dict[str, Any]] = state.get("similar_chapters", []) or []
    if similar:
        lines = ["【相关前文章节】"]
        for s in similar[:5]:
            ch_num = s.get("chapter", "?")
            score = s.get("score", 0)
            lines.append(f"  第{ch_num}章 (相似度={score:.2f})")
        parts.append("\n".join(lines))

    # 4. 角色网络
    network: list[str] = state.get("character_network", []) or []
    if network:
        parts.append(f"【故事中已有角色】{'、'.join(network[:25])}")

    # 5. 滑动窗口（仅保留 Layer 2-3，剔除 Layer 1「前情提要」）
    # v7.5+: DB 前情提要 = raw text truncation（text[:1000]），质量极低。
    # Layer 1 的功能已由 writer.py 的 runtime 版本替代：
    #   【近期章节回顾】— 从 completed_chapters 构建，含 LLM 结构摘要，
    #   每章 250 字 × 最近 4 章，质量远高于 DB 截断。
    #   重要事件由「关键历史事件」覆盖，角色状态由「跨章角色状态追踪」覆盖。
    sw: str = state.get("sliding_window", "") or ""
    if sw:
        # 剔除「前情提要」段（从开头到下一个「」或末尾）
        if "【前情提要】" in sw:
            pos = sw.find("【前情提要】")
            next_section = sw.find("【", pos + 6)  # 跳过当前「」
            if next_section > pos:
                parts.append(sw[next_section:])  # 仅追加【当前卷】+【关键历史事件】
            # 如果没有后续「」段，则整个 sliding_window 只有「前情提要」→ 跳过
        else:
            parts.append(sw)

    # 6. 弧线状态
    arc: str = state.get("arc_status", "") or ""
    if arc:
        parts.append(arc)

    # 7. Phase2
    for key, label in [
        ("audit_status", "一致性审计"),
        ("foreshadowing_status", "伏笔管理"),
        ("pacing_status", "节奏分析"),
    ]:
        val: Any = state.get(key, "") or ""
        if val and val.strip():
            parts.append(val)

    # 8. Phase3
    for key in ["recovery_status", "cost_status", "quality_status", "volume_status"]:
        val = state.get(key, "") or ""
        if val and val.strip():
            parts.append(val)

    # v7.3-fix: 兜底 — 所有上下文为空时至少返回章号/项目名
    # 避免 writer 在真空中写作导致上下文断裂
    if not parts:
        ch = state.get("chapter_number", 1)
        project: str = state.get("project_name", "")
        parts.append(
            f"【上下文】（数据库不可用，使用基础信息）\n"
            f"项目：{project} | 当前章节：第{ch}章"
        )
        logger.warning(
            "[context_builder] 所有上下文为空，使用兜底信息 ch=%d project=%s",
            ch,
            project,
        )

    return {"writer_context": "\n\n".join(parts)}


# ── Graph Builder ──────────────────────────────────────────────────────────────


def build_context_builder() -> CompiledStateGraph:
    """构建 ContextBuilder 子图。

    Returns:
        编译后的 StateGraph，可作为 writing_crew 的子图节点。
    """
    builder = StateGraph(ContextBuilderState)

    builder.add_node("load_state", _load_state_node)
    builder.add_node("build_context", _build_context_node)
    builder.add_node("aggregate", _aggregate_node)

    builder.add_edge(START, "load_state")
    builder.add_edge("load_state", "build_context")
    builder.add_edge("build_context", "aggregate")
    builder.add_edge("aggregate", END)

    return builder.compile()
