"""VerdictEngine — 统一 LLM 评审融合引擎（v8.2）。

将统一评审结果（UnifiedReviewResult）融合为 VerdictResult：
LLM 综合分 + 迭代加分 → 路由三态（PASS/REFINE/REWRITE）。

v8.2: 清除程序化传感器 / 5 LLM 维度 / 多轮辩论 / 加权融合 / 校准，
评分职责 100% 由统一 LLM 评审（evaluation/unified）承担。

不是子图，不需要 LangGraph 编排。VerdictEngine 节点内部调用。
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from novelfactory.agents.infra import get_logger
from novelfactory.config.constants import (
    VERDICT_ITERATION_BONUS_MAX,
    VERDICT_ITERATION_BONUS_REFINE,
    VERDICT_ITERATION_BONUS_REWRITE,
    VERDICT_PASS_THRESHOLD,
    VERDICT_REFINE_THRESHOLD,
)
from novelfactory.config.quality_params import get_param
from novelfactory.evaluation.schemas import (
    AttemptInfo,
    FeedbackBundle,
    VerdictLevel,
    VerdictResult,
)

logger = get_logger(__name__)


class VerdictEngine:
    """统一 LLM 评审融合引擎（v8.2）。

    融合逻辑：
        1. UnifiedReviewEngine 单次调用五视角评审 → UnifiedReviewResult
           （综合分 + 四维分项 + 毒点/爽点/水段 + 跨章分 + 分歧点）
        2. 自洽校验（severe 毒点/无爽点硬约束）在 unified/parser 内完成
        3. 严重分歧 → 轻量仲裁（unified/arbitration）
        4. _fuse_unified：LLM 综合分 + 迭代宽松加分 → VerdictResult

    决策规则（3 级）：
        1. 双向次数用尽 → PASS（防死循环兜底）
        2. 评审失败或 LLM severe 毒点（未用尽重写）→ REWRITE
        3. final >= pass_threshold → PASS；>= refine_threshold → REFINE；否则 REWRITE

    v8.2 已清除：程序化传感器 / 5 LLM 评分维度 / 多轮辩论 / 加权融合 / 校准模块。
    """

    def __init__(self) -> None:
        pass

    async def evaluate(
        self,
        chapter_text: str,
        genre: str,
        genre_scoring_guide: str,
        prev_summary: str,
        chapter_index: int,
        attempt_info: AttemptInfo,
        reviewer_llm: BaseChatModel,
        debate_llm: BaseChatModel,
    ) -> VerdictResult:
        """执行统一 LLM 评审，返回统一决议（async）。

        v8.2: 单次 LLM 调用完成五视角评审（老书虫/番茄编辑/读者/评论员/四维分项），
        替代旧"程序化传感器 + 5 LLM 维度并行 + 多轮辩论"。

        Args:
            chapter_text: 章节文本
            genre: 题材
            genre_scoring_guide: 题材评分指南
            prev_summary: 前文摘要
            chapter_index: 章节序号
            attempt_info: 重写/润色次数追踪
            reviewer_llm: 统一评审 LLM
            debate_llm: 已弃用（v8.2 辩论由分歧仲裁替代），保留签名兼容调用方

        Returns:
            VerdictResult
        """
        logger.info(
            "[VerdictEngine] ch%d 开始评审 | rewrite=%d/%d refine=%d/%d",
            chapter_index,
            attempt_info.loop_count,
            attempt_info.max_rewrite,
            attempt_info.refine_attempts,
            attempt_info.max_refine,
        )

        # v8.2: 统一 LLM 评审 — 单次调用五视角评审（替代程序化传感器 + 5 LLM 维度 + 多轮辩论）
        from novelfactory.evaluation.unified import UnifiedReviewEngine

        unified_engine = UnifiedReviewEngine(reviewer_llm)
        ur = await unified_engine.evaluate(
            chapter_text=chapter_text,
            genre=genre,
            prev_summary=prev_summary,
            guide=genre_scoring_guide,
            retries=int(get_param("unified.max_retries", 1)),
        )

        verdict = self._fuse_unified(
            ur,
            attempt_info,
            chapter_length=len(chapter_text),
        )

        logger.info(
            "[VerdictEngine] ch%d 统一评审 | final=%.1f quality=%.1f toxic=%d shuang=%d water=%d failed=%s retried=%s",
            chapter_index,
            ur.final_score,
            ur.four_dim.total(),
            len(ur.toxic_points),
            ur.shuangdian_count,
            len(ur.water_paragraphs),
            ur.failed,
            ur.retried,
        )

        return verdict

    # ═══════════════════════════════════════════════════════════════════════
    #  v8.2 统一评审融合（替代 _fuse 加权融合 / _decide_level 多分支 / 校准）
    # ═══════════════════════════════════════════════════════════════════════

    def _fuse_unified(
        self,
        ur,
        attempt_info: AttemptInfo,
        *,
        chapter_length: int,
    ) -> VerdictResult:
        """统一评审融合：LLM 综合分 + 迭代加分 + 路由三态（简化版 _fuse）。"""
        final_score = ur.final_score
        # 迭代宽松加分（保留原机制，缓解反复修复；默认值来自 constants）
        if attempt_info.loop_count > 0 or attempt_info.refine_attempts > 0:
            bonus_rewrite = float(
                get_param("verdict.iteration_bonus.rewrite") or VERDICT_ITERATION_BONUS_REWRITE
            )
            bonus_refine = float(
                get_param("verdict.iteration_bonus.refine") or VERDICT_ITERATION_BONUS_REFINE
            )
            bonus_max = float(
                get_param("verdict.iteration_bonus.max") or VERDICT_ITERATION_BONUS_MAX
            )
            bonus = (
                attempt_info.loop_count * bonus_rewrite
                + attempt_info.refine_attempts * bonus_refine
            )
            final_score = min(final_score + bonus, ur.final_score + bonus_max)

        quality_score = ur.four_dim.total()
        feedback = self._build_unified_feedback(ur)
        level = self._decide_unified_level(final_score, ur, attempt_info)
        verdict = VerdictResult(
            level=level,
            passed=level == VerdictLevel.PASS,
            final_score=round(final_score, 1),
            quality_score=quality_score,
            programmatic_score=0.0,  # 程序化已移除，字段保留兼容
            cross_chapter_consistency=ur.cross_chapter_score,
            debate_penalty=0.0,  # 辩论惩罚由统一评审自洽校验承接
            ai_style_score=ur.human_like_score,
            lao_shu_chong_score=ur.final_score,
            # v8.2: LLM 追踪字段透出统一评审分（下游 replay/dashboard 兼容且保留语义）
            llm_semantic_score=ur.final_score,
            llm_human_like_score=round(ur.human_like_score * 100.0, 1),
            llm_attraction_score=ur.attraction_score,
            llm_severe_toxic_detected=ur.severe_toxic,
            llm_analysis_failed=ur.failed,
            feedback=feedback,
            attempt_info=attempt_info,
            has_severe_toxic=ur.severe_toxic,
            is_short_text=chapter_length < 100,
        )
        return verdict

    def _decide_unified_level(
        self,
        final_score: float,
        ur,
        attempt_info: AttemptInfo,
    ) -> VerdictLevel:
        """路由三态（保留阈值与次数兜底，severe 毒点由 LLM 判定驱动强制重写）。"""
        pass_th = float(get_param("verdict.pass_threshold") or VERDICT_PASS_THRESHOLD)
        refine_th = float(get_param("verdict.refine_threshold") or VERDICT_REFINE_THRESHOLD)

        # 双向用尽强制通过（防死循环）
        both_exhausted = attempt_info.rewrite_exhausted and attempt_info.refine_exhausted
        if both_exhausted:
            logger.warning(
                "[VerdictEngine] 双向次数用尽强制通过 rewrite=%d/%d refine=%d/%d final=%.1f",
                attempt_info.loop_count, attempt_info.max_rewrite,
                attempt_info.refine_attempts, attempt_info.max_refine,
                final_score,
            )
            return VerdictLevel.PASS

        # 统一评审失败 → 未耗尽重写时转 REWRITE（防垃圾章假通过）
        if ur.failed and not attempt_info.rewrite_exhausted:
            return VerdictLevel.REWRITE

        # LLM 严重毒点 → 未耗尽重写时强制 REWRITE（替代程序化 severe_toxic 触发）
        if ur.severe_toxic and not attempt_info.rewrite_exhausted:
            return VerdictLevel.REWRITE

        if final_score >= pass_th:
            return VerdictLevel.PASS
        if final_score >= refine_th:
            return VerdictLevel.REFINE
        return VerdictLevel.REWRITE

    def _build_unified_feedback(self, ur) -> FeedbackBundle:
        """从统一评审结果构建反馈包（refiner/writer 消费）。"""
        from novelfactory.evaluation.schemas import FeedbackBundle

        toxic_types = [t.get("type", "") for t in ur.toxic_points if t.get("type")]
        return FeedbackBundle(
            score_summary=(
                f"统一评审：{ur.final_score:.0f}/100 | 四维{ur.four_dim.total():.0f} | "
                f"爽点{ur.shuangdian_count} | 水段{len(ur.water_paragraphs)} | "
                f"跨章{ur.cross_chapter_score:.0f}"
            ),
            review_comments=ur.review_comments,
            toxic_points=toxic_types,
            shuangdian_points=ur.shuangdian_points,
            debate_issues=list(ur.perspective_disagreements),
            debate_strengths=ur.shuangdian_points[:3],
            debate_suggestions="\n".join(ur.fix_suggestions),
        )


