"""Review Service — shared evaluation interface for both batch and conversational modes.

This is the fusion point for the evaluation/review layer:
- Batch mode: VerdictEngine is called from verdict_engine_node in the Writing Crew
- Conversational mode: ReviewAgent calls this service from the Lead Agent graph

Both modes share the same VerdictEngine, CalibrationModule, and FeedbackBuilder.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from novelfactory.config.constants import MAX_REWRITE_ATTEMPTS, VERDICT_REFINE_THRESHOLD
from novelfactory.config.llm import get_reviewer_llm, get_worker_llm
from novelfactory.evaluation.schemas import (
    AttemptInfo,
    FeedbackBundle,
    VerdictLevel,
    VerdictResult,
)
from novelfactory.evaluation.verdict.engine import VerdictEngine

logger = logging.getLogger(__name__)


class ReviewService:
    """Shared review service — callable from both batch and conversational modes.

    Usage (batch mode):
        service = ReviewService()
        verdict = await service.evaluate(chapter_text="...", genre="...", ...)

    Usage (conversational mode):
        service = ReviewService()
        result = await service.evaluate_conversational(chapter_text="...", user_feedback="...")
    """

    def __init__(self) -> None:
        self._engine = VerdictEngine()

    async def evaluate(
        self,
        chapter_text: str,
        genre: str,
        genre_scoring_guide: str = "",
        prev_summary: str = "",
        chapter_index: int = 1,
        loop_count: int = 0,
        refine_attempts: int = 0,
        reviewer_llm: BaseChatModel | None = None,
        debate_llm: BaseChatModel | None = None,
    ) -> VerdictResult:
        """Execute full review pipeline (shared interface).

        This is the same pipeline used by the batch Writing Crew.
        """
        attempt_info = AttemptInfo(
            loop_count=loop_count,
            refine_attempts=refine_attempts,
            max_rewrite=MAX_REWRITE_ATTEMPTS,
            max_refine=2,
        )

        reviewer_llm = reviewer_llm or get_reviewer_llm()
        debate_llm = debate_llm or get_worker_llm()

        try:
            verdict = await self._engine.evaluate(
                chapter_text=chapter_text,
                genre=genre,
                genre_scoring_guide=genre_scoring_guide,
                prev_summary=prev_summary,
                chapter_index=chapter_index,
                attempt_info=attempt_info,
                reviewer_llm=reviewer_llm,
                debate_llm=debate_llm,
            )
            return verdict
        except Exception as e:
            logger.exception("[ReviewService] Evaluation failed: %s", e)
            return VerdictResult(
                level=VerdictLevel.PASS,
                passed=True,
                final_score=VERDICT_REFINE_THRESHOLD,
                quality_score=VERDICT_REFINE_THRESHOLD,
                programmatic_score=0.5,
                cross_chapter_consistency=VERDICT_REFINE_THRESHOLD,
                debate_penalty=0.0,
                feedback=FeedbackBundle(
                    score_summary=f"评审异常降级: {e}",
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

    async def evaluate_conversational(
        self,
        chapter_text: str,
        genre: str = "",
        user_feedback: str = "",
    ) -> dict[str, Any]:
        """Conversational review — returns human-readable review summary.

        This is optimized for the conversational mode, returning a structured
        summary that the ReviewAgent can present to the user.

        Args:
            chapter_text: 待评审的章节文本
            genre: 题材分类（默认空字符串，使用通用评分标准）
            user_feedback: 用户反馈文本。此参数由调用方（ReviewAgent）负责
                在评审 prompt 中注入，本方法不直接传递给 ``evaluate()``。
                ReviewAgent 应将 ``user_feedback`` 拼接到发给 LLM 的评审
                prompt 中，使评审能感知用户的具体关注点和修改意见。

        Returns:
            包含 level、评分、反馈等结构化评审摘要的字典。
        """
        verdict = await self.evaluate(
            chapter_text=chapter_text,
            genre=genre,
        )

        level_text = {
            VerdictLevel.PASS: "✅ 通过",
            VerdictLevel.REFINE: "🔧 需润色",
            VerdictLevel.REWRITE: "🔄 需重写",
        }.get(verdict.level, "❓ 未知")

        return {
            "level": verdict.level.value,
            "level_text": level_text,
            "final_score": verdict.final_score,
            "quality_score": verdict.quality_score,
            "ai_style_score": verdict.ai_style_score,
            "lao_shu_chong_score": verdict.lao_shu_chong_score,
            "cross_chapter_consistency": verdict.cross_chapter_consistency,
            "debate_penalty": verdict.debate_penalty,
            "summary": verdict.feedback.score_summary or "评审完成",
            "toxic_points": verdict.feedback.toxic_points,
            "shuangdian_points": verdict.feedback.shuangdian_points,
            "suggestions": verdict.feedback.debate_suggestions,
            "has_severe_toxic": verdict.has_severe_toxic,
            "attempts": {
                "rewrite_count": verdict.attempt_info.loop_count,
                "refine_count": verdict.attempt_info.refine_attempts,
            },
        }


# Module-level singleton for easy access
_service: ReviewService | None = None


def get_review_service() -> ReviewService:
    """Get the shared ReviewService singleton."""
    global _service
    if _service is None:
        _service = ReviewService()
    return _service