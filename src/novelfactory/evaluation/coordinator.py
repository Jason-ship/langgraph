"""评审协调器 — VerdictEngine 的图节点封装。

替代现有的 reviewer.py _chapter_reviewer_node。
作为 writing_crew 图中的 verdict_engine 节点。

职责：
    1. 从 state 读取章节文本、上下文、次数信息
    2. 调用 VerdictEngine.evaluate() 执行完整评审
    3. 将 VerdictResult 写回 state（含向后兼容字段）
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from novelfactory.agents.infra import get_crew_stream, get_logger
from novelfactory.config.constants import (
    MAX_REWRITE_ATTEMPTS,
    VERDICT_REFINE_THRESHOLD,
)
from novelfactory.config.constants import (
    get_genre_thresholds as _ggt,
)
from novelfactory.config.constants import (
    resolve_genre as _resolve_genre,
)
from novelfactory.config.llm import get_reviewer_llm, get_worker_llm
from novelfactory.evaluation.schemas import (
    AttemptInfo,
    FeedbackBundle,
    VerdictLevel,
    VerdictResult,
)
from novelfactory.evaluation.verdict.engine import VerdictEngine
from novelfactory.utils.wall_time_tracker import WallTimeTracker

logger = get_logger(__name__)

# 默认最大润色次数（按分数段不同，这里取上限）
_DEFAULT_MAX_REFINE = 2


# v6.1: 模块级 WallTimeTracker
_wall_tracker = WallTimeTracker()


async def verdict_engine_node(state: dict[str, Any]) -> dict[str, Any]:
    """verdict_engine 图节点 (async) — 替代 _chapter_reviewer_node。

    执行完整评审流程：
        程序化分析 → 知情辩论 → 四维LLM评分 → 融合计算 → VerdictResult

    Args:
        state: LangGraph 状态

    Returns:
        状态更新字典
    """
    cr = state.get("crew_result", {})
    current_ch = state.get("current_chapter", cr.get("current_chapter_number", 1))
    _wall_tracker.start("verdict_engine", phase="writing", chapter=current_ch)

    loop_count = state.get("loop_count", 0)
    # v6.3-fix: 从 crew_result._rewrite_count 读取备份，修复子图 loop_count 丢失
    if loop_count == 0 and state.get("crew_result", {}).get("_rewrite_count", 0) > 0:
        loop_count = state["crew_result"]["_rewrite_count"]
    # v7.3-fix: 从 crew_result._refine_count 读取备份（与 _rewrite_count 逻辑对称）
    # refiner_node 写入 crew_result._refine_count，但状态合并时可能丢失顶层 refine_attempts
    # v7.8-fix: 只回退丢失值不覆盖重置
    # writer_node 重置 refine_attempts=0 时同时重置 _refine_count=0，
    # 此时 refine_attempts==0 和 _refine_count==0 同时成立，不回退。
    refine_attempts = state.get("refine_attempts", 0)
    if refine_attempts == 0 and cr.get("_refine_count", 0) > 0:
        # 顶层丢失（subgraph 状态传播未合并）→ 从 crew_result 回退
        refine_attempts = cr["_refine_count"]
    prefix = f"chapter_{current_ch}"
    chapter_draft = state.get("chapter_draft", "") or cr.get("chapter_draft", "")

    sw = get_crew_stream("writing", prefix)
    if sw:
        sw.write(f"\n[verdict_engine] 开始审核第{current_ch}章...\n")

    # 构建题材评分指引
    genre = cr.get("genre", "")
    genre_scoring_guide = _build_genre_scoring_guide(genre)

    # 前文摘要（从 crew_result 获取）
    prev_summary = cr.get("prev_chapters_summary", "") or cr.get("world_setting", "")

    # 次数追踪
    attempt_info = AttemptInfo(
        loop_count=loop_count,
        refine_attempts=refine_attempts,
        max_rewrite=MAX_REWRITE_ATTEMPTS,
        max_refine=_DEFAULT_MAX_REFINE,
    )

    # 获取 LLM 实例
    reviewer_llm = get_reviewer_llm()
    debate_llm = get_worker_llm()

    # 执行评审 (async)
    engine = VerdictEngine()
    try:
        verdict = await engine.evaluate(
            chapter_text=chapter_draft,
            genre=genre,
            genre_scoring_guide=genre_scoring_guide,
            prev_summary=prev_summary,
            chapter_index=current_ch,
            attempt_info=attempt_info,
            reviewer_llm=reviewer_llm,
            debate_llm=debate_llm,
        )
    except Exception as e:
        logger.exception("[verdict_engine] 评审失败，降级为默认通过: %s", e)
        if sw:
            sw.write(f"\n[verdict_engine] ⚠ 评审失败，降级为默认通过: {e}\n")

        verdict = VerdictResult(
            level=VerdictLevel.PASS,
            passed=True,
            final_score=VERDICT_REFINE_THRESHOLD,
            quality_score=VERDICT_REFINE_THRESHOLD,
            programmatic_score=50.0,
            cross_chapter_consistency=VERDICT_REFINE_THRESHOLD,
            debate_penalty=0.0,
            feedback=FeedbackBundle(
                summary=f"评审异常降级: {e}",
                toxic_points=[],
                shuangdian_points=[],
                debate_issues=[],
                debate_strengths=[],
                debate_suggestions="",
            ),
            attempt_info=attempt_info,
            has_severe_toxic=False,
            calibration_reason=f"评审失败降级: {e}",
        )

    # 流式输出
    if sw:
        level_text = {
            VerdictLevel.PASS: "通过",
            VerdictLevel.REFINE: "需润色",
            VerdictLevel.REWRITE: "需重写",
        }.get(verdict.level, "未知")

        # v7.3-fix: force-pass 时标记
        force_pass = ""
        if verdict.level == VerdictLevel.PASS and (
            attempt_info.rewrite_exhausted or attempt_info.refine_exhausted
        ):
            force_pass = " ⚠️ 次数用尽强制通过"

        sw.write(
            f"[verdict_engine] 第{current_ch}章评审：{verdict.final_score:.1f}/100 → {level_text}{force_pass}\n"
            f"  四维={verdict.quality_score:.0f} | AI味={verdict.ai_style_score:.2f} | "
            f"老书虫={verdict.lao_shu_chong_score:.0f} | "
            f"跨章={verdict.cross_chapter_consistency:.0f} | "
            f"辩论惩罚={verdict.debate_penalty:.0f}\n"
        )

        if verdict.calibration_reason:
            sw.write(f"  [校准] {verdict.calibration_reason}\n")

        if verdict.feedback.toxic_points:
            sw.write(f"  [毒点] {', '.join(verdict.feedback.toxic_points)}\n")
        if verdict.feedback.shuangdian_points:
            sw.write(f"  [爽点] {', '.join(verdict.feedback.shuangdian_points)}\n")
        if verdict.feedback.debate_issues:
            sw.write(f"  [辩论问题] {len(verdict.feedback.debate_issues)}个\n")

    # 构建状态更新
    state_update = verdict.to_state_dict()

    # v6.1: composite_score → programmatic_score
    state_update["crew_result"] = {
        **cr,
        "review_result": verdict.model_dump(),
        "quality_score": verdict.quality_score,
        "programmatic_score": verdict.programmatic_score,
        "ai_style_score": verdict.ai_style_score,
        "lao_shu_chong_score": verdict.lao_shu_chong_score,
        "final_score": verdict.final_score,
    }

    # 次数更新
    if verdict.level == VerdictLevel.REWRITE:
        state_update["loop_count"] = loop_count + 1
        state_update["refine_attempts"] = 0
        # v6.3-fix: 备份 _rewrite_count 到 crew_result，防止 loop_count
        # 在子图状态合并中丢失。用法同 chapter_refiner 的 _refine_count。
        state_update["crew_result"] = {
            **state_update.get("crew_result", {}),
            "_rewrite_count": loop_count + 1,
        }
    elif verdict.level == VerdictLevel.REFINE:
        state_update["loop_count"] = loop_count
        # Defensive: explicitly pass through refine_attempts so VerdictEngine
        # in the next cycle sees the correct count.  chapter_refiner_node
        # is the authoritative increment site, but if it hasn't run yet
        # (first REFINE pass) this preserves the current value.
        state_update["refine_attempts"] = refine_attempts
    else:  # PASS
        state_update["loop_count"] = loop_count

    state_update["human_guidance"] = state.get("human_guidance", "")

    # v6.1: 记录 WallTime 追踪数据
    _wall_tracker.end("verdict_engine")
    if _wall_tracker.to_dict().get("records"):
        state_update["wall_time_data"] = __import__("json").dumps(
            _wall_tracker.to_dict()
        )

    # Chat UI message
    level_text = {
        VerdictLevel.PASS: "通过",
        VerdictLevel.REFINE: "需润色",
        VerdictLevel.REWRITE: "需重写",
    }.get(verdict.level, "未知")

    state_update["messages"] = [
        AIMessage(
            content=(
                f"第{current_ch}章评审：{verdict.final_score:.0f}/100 "
                f"（四维{verdict.quality_score:.0f} | AI味{verdict.ai_style_score:.2f} | "
                f"老书虫{verdict.lao_shu_chong_score:.0f} | "
                f"跨章{verdict.cross_chapter_consistency:.0f}）→ {level_text}"
            ),
            name="verdict_engine",
        )
    ]

    return state_update


def _build_genre_scoring_guide(genre: str | None) -> str:
    """根据题材生成 LLM 评分指引。

    迁移自 reviewer.py _build_genre_scoring_guide。
    """
    if not genre:
        return ""

    resolved = _resolve_genre(genre)
    if resolved == "default":
        resolved = genre

    # 优先从 SkillLoader 加载
    try:
        from novelfactory.skills.loader import SkillLoader

        loader = SkillLoader()
        loader.discover()
        skills = loader.get_skills_by_genre(resolved)
        if skills:
            return f"▓▓▓ 题材感知：{resolved}（来自 SKILL.md）▓▓▓\n{skills[0].body}"
    except Exception as e:
        logger.debug("[genre_guide] SkillLoader 不可用: %s", e)

    # 无 Skill 文件时从 GENRE_THRESHOLDS 生成通用指引
    gt = _ggt(genre)
    desc = gt.get("description", "")
    themes = gt.get("themes", [])
    q_threshold = gt.get("quality_score", 85)
    ai_threshold = gt.get("ai_style", 0.30)

    parts = [f"▓▓▓ 题材感知：{resolved} ▓▓▓"]
    if desc:
        parts.append(f"本文特点：{desc}")
    if themes:
        parts.append(f"常见元素：{'、'.join(themes)}")
    parts.append(
        f"评分建议：四维评分≥{int(q_threshold)}分视为通过，AI味指数≤{ai_threshold}为合格。"
    )
    return "\n".join(parts)
