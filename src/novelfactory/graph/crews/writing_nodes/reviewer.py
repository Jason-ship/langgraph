"""Chapter refiner node extracted from writing_crew.py.

Provides:
    - _chapter_refiner_node   — refines chapter based on review feedback

Notes:
    v6.3: _chapter_reviewer_node was replaced by verdict_engine_node
    (evaluation.coordinator.verdict_engine_node).  _merge_debate_outputs
    and the duplicate _build_genre_scoring_guide are also removed.
    Only _chapter_refiner_node remains in this module.
"""

from __future__ import annotations

from typing import Any

from novelfactory.agents.infra import get_crew_stream, get_logger, read_usage_tracking
from novelfactory.agents.writing_agents import create_chapter_refiner_agent
from novelfactory.config.llm import get_worker_llm
from novelfactory.evaluation.utils import normalize_paragraph_refs
from novelfactory.state.crew_state import BaseCrewState

logger = get_logger(__name__)


# ── Node: chapter_refiner ─────────────────────────────────────────────────────


async def _chapter_refiner_node(state: BaseCrewState) -> dict[str, Any]:
    """Refine chapter based on review feedback (async).

    v7.0: 定向段落修复 — 从 unified verdict_result 读取反馈，
          refiner agent 只输出需要修改的段落，程序化合并回原文。
    """
    cr: dict[str, Any] = state.get("crew_result", {})
    current_ch: int = int(  # type: ignore[call-overload]
        state.get("current_chapter", cr.get("current_chapter_number", 1))
    )
    prefix = f"chapter_{current_ch}"
    chapter_draft: str = (
        state.get("chapter_draft", "") or cr.get("chapter_draft", "") or ""  # type: ignore[assignment]
    )
    review_result: Any = cr.get("review_result", {})

    # ── Read feedback from unified verdict_result ─────────────────────────
    # feedback is the VerdictResult.feedback (FeedbackBundle) dict
    feedback = {}
    if isinstance(review_result, dict):
        feedback = review_result.get("feedback", {})
        if not isinstance(feedback, dict):
            feedback = {}

    # Helper to extract feedback fields, with fallback to flat state
    def _fb(key: str, default: Any = "") -> Any:
        val = feedback.get(key)
        if val is not None and val:
            return val
        # Fallback to flat review_result fields (old format)
        flat_val = review_result.get(key)
        if flat_val is not None and flat_val:
            return flat_val
        # Fallback to top-level state (legacy)
        return state.get(key, default)

    sw = get_crew_stream("writing", prefix)
    if sw:
        sw.write(f"\n[chapter_refiner] 开始定向修复第{current_ch}章...\n")
        ai_fix = _fb("ai_style_fix", "")
        if ai_fix and ai_fix not in ("AI味指数合格，无需特别修改。", ""):
            sw.write(f"[AI味] {ai_fix[:200]}...\n")
        lao_fix = _fb("lao_shu_chong_fix", "")
        if lao_fix and lao_fix not in ("老书虫视角评分良好，保持当前方向。", ""):
            sw.write(f"[老书虫] {lao_fix[:200]}...\n")
        toxic = _fb("toxic_points", [])
        if toxic:
            sw.write(f"[毒点] {'、'.join(toxic[:3])}\n")

    # ── Build enhanced review from unified feedback ───────────────────────
    enhanced_review = {
        **review_result,
        "ai_style_fix": _fb("ai_style_fix", ""),
        "lao_shu_chong_fix": _fb("lao_shu_chong_fix", ""),
        "guide_references": state.get("guide_references", []),
        "toxic_points": _fb("toxic_points", []),
        "shuangdian_points": _fb("shuangdian_points", []),
        "debate_issues": _fb("debate_issues", []),
        "debate_strengths": _fb("debate_strengths", []),
        "debate_suggestions": _fb("debate_suggestions", ""),
        "ai_style_metrics_brief": _fb("ai_style_metrics_brief", ""),
        "cross_chapter_brief": _fb("cross_chapter_brief", ""),
        "debate_transcript": _fb("debate_transcript", ""),
    }

    # v7.0: 归一化任何 "第N段" 引用为 [Pi]，确保与 refiner 的段落编号一致
    for text_field in (
        "review_comments",
        "ai_style_fix",
        "lao_shu_chong_fix",
        "debate_suggestions",
        "ai_style_metrics_brief",
        "cross_chapter_brief",
        "debate_transcript",
    ):
        val = enhanced_review.get(text_field)
        if isinstance(val, str) and val:
            enhanced_review[text_field] = normalize_paragraph_refs(val)

    for list_field in ("debate_issues", "toxic_points", "shuangdian_points"):
        vals = enhanced_review.get(list_field)
        if isinstance(vals, list):
            enhanced_review[list_field] = [
                normalize_paragraph_refs(str(v))
                if not isinstance(v, str)
                else normalize_paragraph_refs(v)
                for v in vals
            ]

    refiner_input = {
        "crew_result": {
            "chapter_draft": chapter_draft,
            "review_result": enhanced_review,
            "current_chapter_number": current_ch,
        }
    }

    refiner_agent = create_chapter_refiner_agent(get_worker_llm())

    # v7.8-fix: 短文本自动重试 — 和 chapter_writer 同样的重试保护。
    # 最多重试 2 次（共 3 次尝试），每次注入更强的"必须完整输出"指令。
    # v7.8: async — 使用 agent.ainvoke 避免阻塞事件循环。
    min_refiner_length = 500  # 润色输出至少 500 字才有意义
    refined_chapter = ""
    for attempt in range(1, 4):
        result = await refiner_agent.ainvoke(refiner_input)
        refined_cr = result.get("crew_result", {})
        refined_chapter = refined_cr.get("refined_chapter", "")
        text_len = len(refined_chapter)

        if text_len >= min_refiner_length:
            break

        logger.warning(
            "[chapter_refiner] ch%d 第%d次尝试输出过短 (%d字 < %d字)，即将重试",
            current_ch,
            attempt,
            text_len,
            min_refiner_length,
        )
        if sw:
            sw.write(
                f"[chapter_refiner] ⚠ 第{attempt}次输出过短 ({text_len}字)，"
                f"重试中...\n"
            )
        # 注入更强指令
        hint = (
            f"\n\n[系统警告] 第{attempt}次润色输出仅{text_len}字，内容为空或过短。"
            f"你必须输出至少{min_refiner_length}字的修复后章节正文，"
            f"包含修改后的段落。这是第{attempt + 1}次尝试，请认真完成。"
        )
        refiner_input["crew_result"]["chapter_draft"] = (
            refiner_input["crew_result"].get("chapter_draft", "") + hint
        )

    # 重试后仍为空 → 安全 fallback 原文
    if not refined_chapter or len(refined_chapter) < min_refiner_length:
        refined_chapter = chapter_draft
        logger.info(
            "Chapter %d: using full-text fallback for refiner output",
            current_ch,
        )

    text_len = len(refined_chapter)
    if sw:
        para_count = len(chapter_draft.split("\n\n"))
        sw.write(
            f"[chapter_refiner] 第{current_ch}章修复完成 ({text_len} 字, "
            f"{para_count} 段落)\n"
            f"预览：{refined_chapter[:300].replace(chr(10), '  ')}...\n"
        )

    # v19 P0 #10 fix: also surface accumulated token usage into nested crew_result
    # so that the Command.PARENT update from _exit_for_chapter propagates
    # total_usage to the root state. Without this dual-write, root state's
    # total_usage would be empty — breaking /usage endpoint reports.
    chapter_usage_so_far = read_usage_tracking()
    chapter_record_refine = {
        "chapter_number": current_ch,
        "phase": "writing_refine",
        "prompt_tokens": chapter_usage_so_far.get("prompt_tokens", 0),
        "completion_tokens": chapter_usage_so_far.get("completion_tokens", 0),
        "total_tokens": chapter_usage_so_far.get("total_tokens", 0),
        "estimated_cost_cny": chapter_usage_so_far.get("estimated_cost_cny", 0.0),
        "model_breakdown": chapter_usage_so_far.get("model_breakdown", {}),
        "quality_score": cr.get("review_result", {}).get("quality_score", 0.0),
    }
    # v6.2 FIX: crew_result._refine_count — 以 crew_result 为权威来源
    # 避免顶层 refine_attempts 字段在子图状态传播中丢失计数。
    prev_count = cr.get("_refine_count", 0)
    return {
        "crew_result": {
            **cr,
            "refined_chapter": refined_chapter,
            "chapter_draft": refined_chapter,
            "_refine_count": prev_count + 1,
            "total_usage": {
                "chapter_usages": [chapter_record_refine],
                "prompt_tokens": chapter_usage_so_far.get("prompt_tokens", 0),
                "completion_tokens": chapter_usage_so_far.get("completion_tokens", 0),
                "total_tokens": chapter_usage_so_far.get("total_tokens", 0),
                "estimated_cost_cny": chapter_usage_so_far.get(
                    "estimated_cost_cny", 0.0
                ),
                "model_breakdown": chapter_usage_so_far.get("model_breakdown", {}),
            },
        },
        "chapter_draft": refined_chapter,  # Top-level for reducer
        "quality_score": 0.0,
        "refine_attempts": int(state.get("refine_attempts", 0)) + 1,  # type: ignore[call-overload]
        "human_guidance": state.get("human_guidance", ""),  # persist across nodes
        # v6.1: 不再需要重置 composite_score（已移除）
        "ai_style_score": 0.0,
        "lao_shu_chong_score": 0.0,
        "ai_style_fix": "",
        "lao_shu_chong_fix": "",
    }
