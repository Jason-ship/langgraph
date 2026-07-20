"""Chapter writer node extracted from writing_crew.py.

Provides _chapter_writer_node — writes the initial chapter draft.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from novelfactory.agents.infra import get_crew_stream, reset_usage_tracking
from novelfactory.agents.writing_agents import create_chapter_writer_agent
from novelfactory.config.constants import (
    FALLBACK_TARGET_CHAPTERS,
    MIN_CHAPTER_TEXT_LENGTH,
)
from novelfactory.config.llm import get_worker_llm
from novelfactory.graph.crews.writing_nodes.helpers import _sanitize_human_guidance
from novelfactory.state.crew_state import BaseCrewState

logger = logging.getLogger(__name__)


# ── Node: chapter_writer ────────────────────────────────────────────────────────


def _format_chapter_plan(plan: dict) -> str:
    """Format chapter plan as writer prompt section."""
    scenes = plan.get("scenes", [])
    parts = ["【本章写作计划 — 必须严格遵循】"]
    parts.append(f"标题：{plan.get('title', '')}")
    parts.append(f"核心情节点：{plan.get('core_plot_point', '')}")
    parts.append(f"主视角：{plan.get('pov_character', '')}")
    parts.append(f"情感弧线：{plan.get('emotional_arc', '')}")
    parts.append(f"目标字数：{plan.get('target_word_count', 0)}字")
    if plan.get("foreshadowing_plant"):
        parts.append(f"需埋伏笔：{'，'.join(plan['foreshadowing_plant'])}")
    if plan.get("foreshadowing_resolve"):
        parts.append(f"需回收伏笔：{'，'.join(plan['foreshadowing_resolve'])}")
    if scenes:
        parts.append("\n场景安排：")
        for s in scenes:
            parts.append(
                f"  场景{s.get('scene_number', '?')}：{s.get('purpose', '')}"
                f"（{s.get('location', '')}，{s.get('pov_character', '')}视角）"
            )
    if plan.get("cliffhanger"):
        parts.append(f"结尾悬念：{plan['cliffhanger']}")
    return "\n".join(parts)


async def _chapter_writer_node(state: BaseCrewState) -> dict[str, Any]:
    """Write chapter draft (async). Returns plain dict — routing done by verdict_router."""
    cr: dict[str, Any] = state.get("crew_result", {})
    # v5.9 P0-fix: Send 并行分发时 state.current_chapter 为权威来源，
    # crew_result.current_chapter_number 仅作 fallback（顺序写作兼容）
    current_ch: int = state.get("current_chapter", cr.get("current_chapter_number", 1))  # type: ignore[assignment]
    target: int = (
        cr.get("target_chapters")
        or state.get("target_chapters")  # type: ignore[assignment]
        or FALLBACK_TARGET_CHAPTERS
    )

    # Token-usage tracking: reset at the start of each chapter attempt.
    # Each new chapter attempt (including rewrite loops) starts a fresh usage window.
    reset_usage_tracking()

    # Use module-level StreamWriter (never stored in state)
    prefix = f"chapter_{current_ch}"
    sw = get_crew_stream("writing", prefix)

    sw.section(f"第{current_ch}章 - 创作中")
    sw.write(f"[chapter_writer] 开始撰写第{current_ch}章...\n")

    completed = cr.get("completed_chapters", [])
    # v7.5-fix: 解除 [:300] 截断。chapter_summary 已包含结构化内容
    # （LLM摘要 + 【本章结尾】分隔符 + 最后2段原文），
    # writer 需要看到完整的结尾段落才能做好场景衔接。
    previous_summary = (
        completed[-1].get("chapter_summary", "")[:2000] if completed else ""
    )

    # ── Native LangGraph: ContextBuilder subgraph ──────────────────────
    # Replaces NovelStateTracker.before_chapter() with a compiled subgraph
    # that loads state from PG/Milvus/Neo4j and builds layered context
    # (outline + sliding window + arcs + Phase2/3 status).
    state_prompt: str = state.get("writer_context", "")  # type: ignore[assignment]

    # v7.3-fix: ContextBuilder 返回空时回退到 world_setting/chapter_outlines
    # 避免数据库不可用时 writer 在真空中写作导致上下文断裂
    # 同时兜底顶层 state（crew_result 可能被意外清空）
    if not state_prompt or len(state_prompt.strip()) < 20:
        fallback_parts = []
        # 优先 crew_result，兜底顶层 state
        ws: str = cr.get("world_setting") or state.get("world_setting") or ""  # type: ignore[assignment]
        if ws:
            fallback_parts.append(f"【世界观设定（兜底）】\n{ws[:2000]}")
        co: str = cr.get("chapter_outlines") or state.get("chapter_outlines") or ""  # type: ignore[assignment]
        if co:
            fallback_parts.append(f"【章节大纲（兜底）】\n{co[:2000]}")
        cs: str = cr.get("character_setting") or state.get("character_setting") or ""  # type: ignore[assignment]
        if cs:
            fallback_parts.append(f"【角色设定（兜底）】\n{cs[:1000]}")
        so: str = cr.get("story_outline") or state.get("story_outline") or ""  # type: ignore[assignment]
        if so:
            fallback_parts.append(f"【主线大纲（兜底）】\n{so[:1000]}")
        if fallback_parts:
            state_prompt = "\n\n".join(fallback_parts)
            logger.warning(
                "[chapter_writer] ContextBuilder 返回空，使用 world_setting/chapter_outlines 兜底 "
                "ch=%d fallback_parts=%d",
                current_ch,
                len(fallback_parts),
            )

    # v7.5-fix: 将前章结尾原文注入 state_prompt（与【前章摘要】互补）
    # previous_summary 包含 LLM 摘要 + 结尾原文，但 state_prompt
    # （= writer_context from ContextBuilder）不含结尾原文。
    # 直接追加到 state_prompt，让 ContextBuilder 的丰富上下文
    # 和前章结尾原文并存于同一信源。
    if previous_summary and "【本章结尾】" in previous_summary:
        _parts = previous_summary.split("【本章结尾】", 1)
        _ending = _parts[1].strip() if len(_parts) > 1 else ""
        if _ending:
            state_prompt = (
                (state_prompt + "\n\n" if state_prompt else "")
                + f"【前章结尾场景】\n{_ending}"
            )

    # v7.5+: 放宽压缩 — 从 completed_chapters 构建多章剧情回顾
    # 替代已删除的 DB Layer 1（低质量截断），使用 runtime 高质数据。
    # 展示最近 4 章的摘要（不含结尾原文，避免与【前章结尾场景】重复），
    # 让 writer 感知最近 N 章的情节流动，而非仅上一章。
    if len(completed) >= 2:
        _recent_parts: list[str] = []
        for i in range(min(6, len(completed))):
            ch = completed[-(i + 1)]
            ch_num = ch.get("chapter_number", "?")
            ch_summary = ch.get("chapter_summary", "")
            # 提取摘要部分（去掉【本章结尾】之后的原文）
            if "【本章结尾】" in ch_summary:
                ch_summary = ch_summary.split("【本章结尾】")[0].strip()
            if ch_summary:
                _recent_parts.append(f"  第{ch_num}章：{ch_summary[:300]}")
        if _recent_parts:
            _recent_text = "【近期章节回顾】\n" + "\n".join(reversed(_recent_parts))
            state_prompt = (
                (state_prompt + "\n\n" if state_prompt else "") + _recent_text
            )

    # Sanitize human guidance to prevent prompt injection
    raw_guidance: str = state.get("human_guidance", "")  # type: ignore[assignment]
    safe_guidance = _sanitize_human_guidance(raw_guidance)

    # v4.1: Inject auto_guidance from post_sync_check (volume/quality/foreshadowing)
    auto_guidance = cr.get("auto_guidance", "")
    if auto_guidance:
        state_prompt = (
            (state_prompt + "\n\n" + auto_guidance) if state_prompt else auto_guidance
        )
        logger.info("[chapter_writer] Auto-guidance injected for ch%d", current_ch)

    # v6.3: Inject chapter_plan from chapter_planner node
    chapter_plan: dict[str, Any] = cr.get("chapter_plan") or state.get(
        "chapter_plan", {}
    )  # type: ignore[assignment]
    if chapter_plan and isinstance(chapter_plan, dict) and chapter_plan.get("scenes"):
        plan_text = _format_chapter_plan(chapter_plan)
        state_prompt = (
            (state_prompt + "\n\n" + plan_text) if state_prompt else plan_text
        )
        logger.info(
            "[chapter_writer] Chapter plan injected for ch%d: %d scenes",
            current_ch,
            len(chapter_plan.get("scenes", [])),
        )

    # v6.0 架构优化: 移除 chapter_outlines/world_setting（已由 ContextBuilder 的
    # writer_context 覆盖）。保留 story_outline（主线大纲+金手指体系设计）和
    # character_setting（角色性格/说话方式设计），因为 ContextBuilder 只提供
    # 跨章角色状态追踪（current state），不含角色设计文档。
    writer_input = {
        "crew_result": {
            "story_outline": cr.get("story_outline", ""),
            "character_setting": cr.get("character_setting", ""),
            "current_chapter_number": current_ch,
            "previous_chapter_summary": previous_summary,
            "cross_chapter_state": state_prompt,  # Injected by ContextBuilder subgraph
            # human_guidance: user-provided revision guidance from multi-turn interrupt
            "human_guidance": safe_guidance,
            # v5.11: 传递上一轮评审反馈（重写路径注入）
            "review_result": cr.get("review_result", {}),
            "loop_count": state.get("loop_count", 0),
            "refine_attempts": state.get("refine_attempts", 0),
        }
    }

    writer_agent = create_chapter_writer_agent(get_worker_llm())
    chapter_draft = ""
    # v6.0 P0-fix: 短文本自动重试 — LLM API 降级/挂掉/解析失败时重试。
    # 最多重试 2 次（共 3 次尝试），每次注入更强的"必须输出完整章节"指令。
    # v7.8: async — 使用 agent.ainvoke 避免阻塞事件循环。
    for attempt in range(1, 4):  # 1-based for logging clarity
        result = await writer_agent.ainvoke(writer_input)
        chapter_draft = result.get("crew_result", {}).get("chapter_draft", "")
        text_len = len(chapter_draft)

        if text_len >= MIN_CHAPTER_TEXT_LENGTH:
            break  # 有完整章节，正常退出

        logger.warning(
            "[chapter_writer] ch%d 第%d次尝试生成文本过短 (%d字 < %d字)，即将重试",
            current_ch,
            attempt,
            text_len,
            MIN_CHAPTER_TEXT_LENGTH,
        )
        sw.write(
            f"[chapter_writer] ⚠ 第{attempt}次尝试文本过短 ({text_len}字)，重试中...\n"
        )
        # 每次重试追加更强指令，强制 LLM 输出完整内容
        hint = (
            f"\n\n[系统警告] 第{attempt}次输出仅{text_len}字，完全不符合要求。"
            f"你必须输出至少{MIN_CHAPTER_TEXT_LENGTH}字的完整章节正文，包含场景、对话、情节推进。"
            f"这是第{attempt + 1}次尝试，请认真完成。"
        )
        # 注入到 writer_input
        writer_input["crew_result"]["cross_chapter_state"] = (
            writer_input["crew_result"].get("cross_chapter_state", "") + hint
        )

    text_len = len(chapter_draft)
    sw.write(f"[chapter_writer] 完成第{current_ch}章草稿 ({text_len} 字)\n")
    if text_len < MIN_CHAPTER_TEXT_LENGTH:
        sw.write(
            f"[chapter_writer] ⚠ 重试后文本仍过短 ({text_len}字 < {MIN_CHAPTER_TEXT_LENGTH}字)\n"
        )
    # Write first 500 chars as preview
    preview = chapter_draft[:MIN_CHAPTER_TEXT_LENGTH].replace("\n", "  ") + (
        "..." if text_len > MIN_CHAPTER_TEXT_LENGTH else ""
    )
    sw.write(f"预览：{preview}\n")

    return {
        "crew_result": {
            **cr,
            "chapter_draft": chapter_draft,
            "current_chapter_number": current_ch,
            "target_chapters": target,
            "human_guidance": state.get(
                "human_guidance", ""
            ),  # persist guidance across crew invocations
            # v7.8-fix: 重置 _refine_count，避免 coordinator 回退读到旧值
            "_refine_count": 0,
        },
        "chapter_draft": chapter_draft,  # Top-level for reducer
        # v6.2 FIX (R3): 透传 loop_count 而非重置为 0。
        # 此前每次 chapter_writer 运行都清零 loop_count，导致 reviewer 的
        # loop_count++ 永远无法突破 MAX_REWRITE_ATTEMPTS 上限保护，
        # 形成 writer→reviewer→score<60→writer 死循环。
        # 新章节的 loop_count 由 prepare_writing 重置，重写路径沿用现有计数。
        "loop_count": state.get("loop_count", 0),
        "refine_attempts": 0,  # Reset refine budget when starting a new chapter or rewrite
        "quality_score": 0.0,
        "human_guidance": state.get(
            "human_guidance", ""
        ),  # multi-turn guidance context
        # v5.0 新增字段初始化
        "ai_style_score": 0.0,
        "lao_shu_chong_score": 0.0,
        "toxic_points": [],
        "shuangdian_points": [],
        "guide_references": [],
        "ai_style_fix": "",
        "lao_shu_chong_fix": "",
        # ── Chat UI message ──
        "messages": [
            AIMessage(
                content=f"正在创作第{current_ch}章...（目标{target}章）",
                name="chapter_writer",
            )
        ],
    }
