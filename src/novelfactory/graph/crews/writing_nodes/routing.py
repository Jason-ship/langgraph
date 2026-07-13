"""Routing and exit nodes extracted from writing_crew.py.

Provides:
    - _exit_for_chapter    — exit node that decides next parent step

Notes:
    v6.3: _score_router was replaced by verdict_router
    (evaluation.verdict.router.verdict_router).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from novelfactory.agents.infra import (
    cleanup_crew_stream,
    get_crew_stream,
    get_logger,
    read_usage_tracking,
)
from novelfactory.config.constants import (
    MAX_REWRITE_ATTEMPTS,
)
from novelfactory.graph.crews.writing_nodes.helpers import _make_record

logger = get_logger(__name__)


# ── Exit Node ────────────────────────────────────────────────────────────────


def _exit_for_chapter(state: dict) -> dict:
    """Exit Writing Crew after a chapter is approved.

    This is the ONLY crew-exit point.  It increments current_chapter and
    returns completed_chapters to the parent graph.  The parent graph's
    supervisor (main_supervisor_node) handles routing to the next phase:
      • current_chapter < target_chapters  →  back to writing
      • current_chapter >= target_chapters →  sync phase
      • chapter_needs_guidance              →  human-in-the-loop

    Also accumulates the chapter record into completed_chapters.
    Closes the StreamWriter and writes the final chapter text.
    """
    from novelfactory.config.constants import FALLBACK_DEGRADE_RESULT as _FDR
    from novelfactory.config.constants import get_genre_thresholds as _ggt

    cr = state.get("crew_result", {})
    genre = cr.get("genre", "")
    gt = _ggt(genre)
    quality_excellent_dflt = float(gt.get("quality_score", 85.0))
    poor_threshold_dflt = max(55.0, quality_excellent_dflt - 30.0)
    max_rewrite_dflt = MAX_REWRITE_ATTEMPTS

    # v5.9 P0-fix: Send 并行分发时 state.current_chapter 为权威来源
    current_ch = state.get("current_chapter", cr.get("current_chapter_number", 1))
    # Read from top-level state (set by chapter_writer or chapter_refiner)
    chapter_text = state.get("chapter_draft", "") or cr.get(
        "refined_chapter", cr.get("chapter_draft", "")
    )
    # v5.8: review_result 是评分唯一来源（程序化分析权威），无 schema 优先层
    review_result = cr.get("review_result", {})
    quality = float(review_result.get("quality_score", quality_excellent_dflt))
    if quality == quality_excellent_dflt and "quality_score" not in review_result:
        # v5.9 FIX: review_result 完全缺失时使用全局 FALLBACK_DEGRADE_RESULT
        quality = float(_FDR["quality_score"])

    # v7.3-fix: 使用融合后 final_score 判断章节质量（quality_score 四维评分常虚高）
    # 参考 Fiction_Eval 论文：LLM 对自己的作品评分偏高
    final_score = float(state.get("final_score", 0.0))
    verdict = state.get("verdict_result", {})
    if not final_score and verdict:
        final_score = float(verdict.get("final_score", 0.0))

    # v7.4-fix: 当最终通过版质量显著低于历史最佳版时，恢复最佳版
    best_quality = state.get("best_version_quality", 0.0)
    best_text = state.get("best_version_text", "")
    if best_quality > quality + 10 and best_text:
        logger.info(
            "[_exit_for_chapter] ch%d 恢复最佳版本: quality=%.1f→%.1f (text=%d字)",
            current_ch, quality, best_quality, len(best_text),
        )
        chapter_text = best_text
        quality = best_quality

    # Final streaming: write complete text + close (uses module-level cache)
    prefix = f"chapter_{current_ch}"
    sw = get_crew_stream("writing", prefix)
    if sw:
        sw.section(f"第{current_ch}章 - 完成")
        # v6.1: 从 verdict_result 读取 programmatic_score
        verdict = state.get("verdict_result", {})
        stream_composite = verdict.get("programmatic_score", 0.0) or state.get(
            "composite_score", 0.0
        )
        sw.write(
            f"最终正文（共 {len(chapter_text)} 字，"
            f"四维评分：{quality:.1f}/100，"
            f"综合指标：{stream_composite:.4f}）：\n\n"
        )
        sw.write(chapter_text)
        sw.write(f"\n\n--- 第{current_ch}章 完 ---\n")
        # Close via module cache
        cleanup_crew_stream("writing", prefix)

    # v6.1: 从 verdict_result 读取 programmatic_score（权威来源）
    # 兜底读取旧字段 composite_score 保证向前兼容
    verdict = state.get("verdict_result", {})
    composite = float(
        verdict.get("programmatic_score", 0.0) or state.get("composite_score", 0.0)
    )
    # ── 用 LLM 生成有意义的章节摘要 + 嵌入结尾原文（供下一章衔接用） ──
    # v7.5-fix: chapter_summary 包含结构化信息：
    #   LLM摘要 + 【本章结尾】分隔符 + 本章最后2段落原文
    # 让下一章的 writer 在创作时能看到具体的承接场景，而非仅有抽象摘要。
    chapter_summary = ""
    try:
        from novelfactory.config.llm import get_reviewer_llm

        summary_llm = get_reviewer_llm()
        summary_prompt = (
            f"请为以下章节生成一个简洁的摘要（200-300字），"
            f"包含：本章核心情节、关键事件、结尾留下的悬念/过渡。\n\n"
            f"章节正文：\n{chapter_text[:4000]}"
        )
        summary_resp = summary_llm.invoke([("user", summary_prompt)])
        summary_text = (
            summary_resp.content[:500]
            if hasattr(summary_resp, "content")
            else str(summary_resp)[:500]
        )

        # 提取本章最后2个自然段落（约400字），供下一章 writer 做具体衔接
        paras = [p.strip() for p in chapter_text.split("\n\n") if p.strip()]
        ending_text = ""
        if len(paras) >= 2:
            ending_text = "\n\n".join(paras[-2:])[:400]
        elif paras:
            ending_text = paras[-1][:400]

        if ending_text:
            chapter_summary = (
                f"{summary_text}\n\n【本章结尾】\n{ending_text}"
            )
        else:
            chapter_summary = summary_text

        logger.info(
            "ch%d 摘要生成完成 (%d字 + 结尾%d字)",
            current_ch,
            len(summary_text),
            len(ending_text),
        )
    except Exception as e:
        logger.warning("ch%d 摘要生成失败，使用截断: %s", current_ch, e)
        chapter_summary = chapter_text[:200].replace("\n", " ").strip()

    new_record = _make_record(
        current_ch,
        chapter_text,
        quality,
        chapter_summary=chapter_summary,
        review_result=cr.get("review_result"),
        composite_score=composite,
    )

    # State extraction and database writing are handled by dedicated
    # LangGraph nodes (state_extractor_node → database_writer_node)
    # that run before _exit_for_chapter in the graph topology.
    # ── Feishu notification (migrated from writing_crew_entry) ──────────
    # ── Token usage snapshot for this chapter attempt ──────────────────────────
    chapter_usage = read_usage_tracking()
    if sw:
        sw.write(
            f"\n[usage] 本章 tokens: prompt={chapter_usage['prompt_tokens']} "
            f"completion={chapter_usage['completion_tokens']} "
            f"total={chapter_usage['total_tokens']} "
            f"cost≈¥{chapter_usage['estimated_cost_cny']}\n"
        )
    logger.info(
        "ch%d tokens: prompt=%d completion=%d total=%d cost≈¥%.4f",
        current_ch,
        chapter_usage["prompt_tokens"],
        chapter_usage["completion_tokens"],
        chapter_usage["total_tokens"],
        chapter_usage["estimated_cost_cny"],
    )

    # Human-in-the-loop: low final_score + all rewrites exhausted → request guidance from user
    # v7.3-fix: 使用 final_score 而非 quality_score，因为四维 LLM 评分常虚高
    # (Fiction_Eval 论文 §6.4: LLM 对自己的作品评分偏高)
    loop_count = state.get("loop_count", 0)
    # final_score 为 0 时降级使用 quality（兼容旧状态）
    effective_score = final_score if final_score > 0 else quality
    needs_guidance = (
        effective_score < poor_threshold_dflt and loop_count >= max_rewrite_dflt
    )

    # Build chapter usage record for this chapter
    chapter_record = {
        "chapter_number": current_ch,
        "prompt_tokens": chapter_usage["prompt_tokens"],
        "completion_tokens": chapter_usage["completion_tokens"],
        "total_tokens": chapter_usage["total_tokens"],
        "estimated_cost_cny": chapter_usage["estimated_cost_cny"],
        "quality_score": quality,
        "model_breakdown": chapter_usage.get("model_breakdown", {}),
    }

    total_usage_dict = {
        "chapter_usages": [chapter_record],
        "prompt_tokens": chapter_usage["prompt_tokens"],
        "completion_tokens": chapter_usage["completion_tokens"],
        "total_tokens": chapter_usage["total_tokens"],
        "estimated_cost_cny": chapter_usage["estimated_cost_cny"],
        "model_breakdown": chapter_usage.get("model_breakdown", {}),
    }

    # P0-fix: return completed_chapters so the operator.add reducer in
    # NovelFactoryState accumulates chapter records across subgraph
    # invocations. Only the new record is returned — the reducer
    # appends it to the existing list automatically.
    #
    # LangGraph 0.2.x: compiled subgraph final state auto-merges to parent.
    # Return plain dict — NOT Command(PARENT) which is unsupported in 0.2.x.
    # Increment current_chapter so parent graph routes to next chapter.
    #
    # v5.4-fix: 将 quality_score + total_usage 写入 crew_result 以便
    # sync_crew 子图通过 BaseCrewState 共享的 crew_result 字段读取。
    # 这两个字段不在 WritingCrewLocalState schema 中，写为 top-level return
    # 会被 LangGraph 子图→父图合并机制静默丢弃；写入 crew_result 则可通过
    # SyncCrewLocalState 继承的 BaseCrewState.crew_result 透传。
    #
    # v6.0.1 P0-fix: 简化 crew_result 嵌套 — 只保留增量/透传字段
    # （project_name, current_chapter_number, completed_chapters, review_result,
    #   feishu_doc_url 等），不再复制评分字段和 total_usage（这些已经作为
    #   顶层字段返回到父图，避免 dual-source 混淆导致 _last_value 覆盖问题）。
    # v6.1: 顶层不再写 composite_score，消费方统一从 verdict_result 读取
    return {
        # v6.0.1: 仅保留透传所需的 crew_result 字段，
        # 评分和 token 数据由顶层字段权威传递（避免 dual-source 覆盖）
        "crew_result": {
            **cr,
            "project_name": cr.get("project_name", ""),
            "current_chapter_number": current_ch,
            "chapter_draft": chapter_text,
            "quality_score": quality,
            "programmatic_score": composite,
            "completed_chapters": cr.get("completed_chapters", []) + [new_record],
            "total_usage": total_usage_dict,
            "review_result": cr.get("review_result", {}),
        },
        "current_chapter": current_ch + 1,
        # ── 评分字段顶层返回（供父图 auto-merge，权威来源） ──
        "quality_score": quality,
        "ai_style_score": state.get("ai_style_score", 0.0),
        "lao_shu_chong_score": state.get("lao_shu_chong_score", 0.0),
        "total_usage": total_usage_dict,
        "chapter_needs_guidance": needs_guidance,
        "completed_chapters": [new_record],
        # v7.4-fix: 清除 best_version 字段，防止泄漏到下一章
        "best_version_quality": 0.0,
        "best_version_text": "",
        # ── Chat UI message ──
        "messages": [
            AIMessage(
                content=f"第{current_ch}章完成（{len(chapter_text)}字，{quality:.0f}分）{'，需要人工指导' if needs_guidance else '，自动进入下一章'}",
                name="chapter_finalizer",
            )
        ],
    }
