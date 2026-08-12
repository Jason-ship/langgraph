"""统一评审引擎（v8.2）。

- evaluate()：一次 LLM 调用五视角评审 → 标签解析 → 自洽校验 → 严重分歧仲裁 → 返回
- quick_recheck()：修复后回归复查（轻量）
- 降级：重试 N 次后 fallback（默认 60=REFINE 档，触发修复后再评审）
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from novelfactory.agents.infra.async_retry import async_llm_call_with_retry
from novelfactory.evaluation.unified.parser import apply_consistency_check, parse_review_output
from novelfactory.evaluation.unified.prompts import (
    build_quick_recheck_prompt,
    build_unified_review_prompt,
)
from novelfactory.evaluation.unified.schemas import UnifiedReviewResult

logger = logging.getLogger(__name__)

_DEFAULT_FALLBACK = 60.0


def _get_unified_param(key: str, default: float) -> float:
    try:
        from novelfactory.config.quality_params import quality_center

        val = quality_center.get(key)
        if val is not None:
            return float(val)
    except Exception:
        pass
    return default


class UnifiedReviewEngine:
    """统一评审引擎：单次 LLM 调用五视角评审 + 解析 + 自洽校验 + 仲裁 + 降级。"""

    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    async def evaluate(
        self,
        *,
        chapter_text: str,
        genre: str,
        prev_summary: str,
        guide: str,
        retries: int = 1,
    ) -> UnifiedReviewResult:
        fallback = _get_unified_param("unified.fallback_score", _DEFAULT_FALLBACK)
        prompt = build_unified_review_prompt(
            chapter_text=chapter_text, genre=genre, prev_summary=prev_summary, guide=guide,
        )
        last_result = UnifiedReviewResult(failed=True, final_score=fallback)
        for attempt in range(retries + 1):
            try:
                response = await async_llm_call_with_retry(
                    self._llm.ainvoke, prompt, step_name="unified_review"
                )
                raw = response.content if hasattr(response, "content") else str(response)
                result = parse_review_output(raw)
                if result is not None:
                    apply_consistency_check(result)
                    # 分歧驱动仲裁（替代多轮辩论）——仅严重分歧触发
                    severe_dsg = [
                        d for d in result.perspective_disagreements
                        if "severe" in d or "严重" in d
                    ]
                    if severe_dsg:
                        from novelfactory.evaluation.unified.arbitration import arbitrate

                        result.final_score = await arbitrate(
                            self._llm,
                            disagreements=severe_dsg,
                            old_score=result.final_score,
                        )
                    result.failed = False
                    result.retried = attempt > 0
                    return result
                last_result = UnifiedReviewResult(failed=True, final_score=fallback, raw_text=raw)
            except Exception as e:
                logger.warning("[unified_review] attempt %d failed: %s", attempt + 1, e)
        last_result.retried = retries > 0
        return last_result

    async def quick_recheck(
        self,
        *,
        chapter_text: str,
        old_issues: list[str],
        old_score: float,
        retries: int = 1,
    ) -> UnifiedReviewResult:
        fallback = _get_unified_param("unified.fallback_score", _DEFAULT_FALLBACK)
        prompt = build_quick_recheck_prompt(
            chapter_text=chapter_text, old_issues=old_issues, old_score=old_score,
        )
        for attempt in range(retries + 1):
            try:
                response = await async_llm_call_with_retry(
                    self._llm.ainvoke, prompt, step_name="unified_recheck"
                )
                raw = response.content if hasattr(response, "content") else str(response)
                result = parse_review_output(raw)
                if result is not None:
                    result.failed = False
                    return result
            except Exception as e:
                logger.warning("[unified_recheck] attempt %d failed: %s", attempt + 1, e)
        return UnifiedReviewResult(failed=True, final_score=fallback)
