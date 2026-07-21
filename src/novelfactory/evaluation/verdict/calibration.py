"""校准模块 — 集中管理评分校准逻辑，替代散落在 reviewer.py / routing.py 的 hack。

所有校准逻辑集中在此，有明确的触发条件和文档记录。
纯代码，零 LLM。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from novelfactory.evaluation.schemas import (
    AttemptInfo,
    CrossChapterSignals,
    DebateReport,
    ProgrammaticReport,
)

if TYPE_CHECKING:
    from novelfactory.agents.infra import async_llm_call_with_retry

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """校准结果。"""

    score: float
    calibrated: bool
    reason: str


class CalibrationModule:
    """评分校准集中管理。

    替代散落在 reviewer.py L290-315 和 routing.py L138-165 的 hack。
    所有校准逻辑集中在此，有明确的触发条件和文档记录。

    校准规则：
        1. LLM虚高校准 — quality≥90 且 programmatic<0.5 → 压到 70+prog×30
        2. 短文本降级 — is_short_text → 以LLM分为主导（0.8权重）
        3. 严重毒点一票否决 — has_severe_toxic → 分数封顶50
    """

    # 校准阈值（从 constants.py 读取，避免硬编码）
    _LLM_VIRTUAL_HIGH_THRESHOLD = 90.0
    _PROGRAMMATIC_LOW_THRESHOLD = 0.5
    _SHORT_TEXT_LLM_WEIGHT = 0.8
    _SEVERE_TOXIC_SCORE_CAP = 50.0

    def calibrate(
        self,
        final_score: float,
        quality_score: float,
        programmatic: ProgrammaticReport,
        debate: DebateReport,
        cross_chapter: CrossChapterSignals,
        attempt_info: AttemptInfo,
        llm_severe_toxic: bool | None = None,
        llm_analysis_healthy: bool = False,
    ) -> CalibrationResult:
        """执行校准，返回 (校准后分数, 是否校准, 原因)。

        v7.7-fix: 新增 llm_severe_toxic/llm_analysis_healthy 参数，
        当 LLM 语义分析功能正常且明确否认毒点时，跳过分数封顶。

        Args:
            final_score: 融合计算后的原始分数
            quality_score: LLM 四维原始分
            programmatic: 程序化分析报告
            debate: 辩论报告
            cross_chapter: 跨章信号
            attempt_info: 次数追踪
            llm_severe_toxic: LLM 老书虫是否检测到严重毒点
            llm_analysis_healthy: LLM 语义分析是否正常运行

        Returns:
            CalibrationResult
        """
        score = final_score
        calibrated = False
        reason = ""

        # 规则1: LLM评分虚高校准
        # DeepSeek V4 Flash 倾向于给满分，当 LLM 分 >= 90 但程序化分 < 0.5 时
        # 将分数压到 70-85 区间，恢复评分区分度
        if (
            quality_score >= self._LLM_VIRTUAL_HIGH_THRESHOLD
            and programmatic.programmatic_score < self._PROGRAMMATIC_LOW_THRESHOLD
        ):
            score = 70.0 + programmatic.programmatic_score * 30.0
            calibrated = True
            reason = (
                f"LLM虚高校准: quality={quality_score:.0f}→{score:.1f} "
                f"(programmatic={programmatic.programmatic_score:.2f})"
            )
            logger.warning("[校准] %s", reason)

        # 规则2: 短文本降级
        # 程序化分析无法执行时，降低程序化权重，以 LLM 分为主
        if programmatic.is_short_text:
            score = quality_score * self._SHORT_TEXT_LLM_WEIGHT + score * (
                1.0 - self._SHORT_TEXT_LLM_WEIGHT
            )
            calibrated = True
            reason = (
                f"短文本降级: 以LLM分主导 "
                f"({quality_score:.0f}×{self._SHORT_TEXT_LLM_WEIGHT} + {final_score:.1f}×{1.0 - self._SHORT_TEXT_LLM_WEIGHT:.1f})"
            )
            logger.info("[校准] %s", reason)

        # 规则3: 严重毒点一票否决
        # 有 NTR/虐主/圣母等严重毒点时，分数不超过 50
        # v7.7-fix: 当 LLM 语义分析正常运转且明确否认毒点时，不封顶。
        # 程序化传感器是关键词匹配，无法区分"剧情驱动的合理冲突"和"恶意虐主"。
        # 信任 LLM 的语义理解能力。
        if programmatic.has_severe_toxic:
            llm_denies_toxic = (
                llm_analysis_healthy
                and llm_severe_toxic is not None
                and not llm_severe_toxic
            )
            if llm_denies_toxic:
                logger.info(
                    "[校准] 程序化毒点被LLM语义分析否认，跳过分数封顶 "
                    "(severe_toxic=%s, llm_severe_toxic=%s)",
                    programmatic.has_severe_toxic,
                    llm_severe_toxic,
                )
            else:
                cap = self._SEVERE_TOXIC_SCORE_CAP
                if score > cap:
                    score = cap
                    calibrated = True
                    reason = (
                        f"严重毒点一票否决: 分数封顶{cap:.0f} "
                        f"({', '.join(programmatic.severe_toxic_types)})"
                    )
                    logger.warning("[校准] %s", reason)

        return CalibrationResult(
            score=max(0.0, min(100.0, score)),
            calibrated=calibrated,
            reason=reason,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  CalibrationRunner — 定期间隔校准（v7.3 新增）
#  参考 DEEVO (Amazon KDD 2025) 的 Elo 收敛 + Neither Valid nor Reliable (2025) 的校准建议。
#  定期用一组标注好的标准章节跑评分漂移检测。
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CalibrationSample:
    """单个校准样本。"""

    chapter_text: str
    baseline_score: float
    chapter_index: int = 0
    genre: str = ""


@dataclass
class CalibrationReport:
    """校准报告。"""

    avg_deviation: float
    max_deviation: float
    sample_count: int
    alarm: bool
    details: list[tuple[int, float, float]]  # (chapter_index, baseline, current)


class CalibrationRunner:
    """评分漂移检测器 — 定期用标准章节检测评分偏移。

    用法:
        runner = CalibrationRunner(reference_set)
        report = runner.run(current_llm)
        if report.alarm:
            logger.warning("评分漂移: 平均偏离 %.1f 分", report.avg_deviation)
    """

    def __init__(self, reference_set: list[CalibrationSample]):
        if not reference_set:
            raise ValueError("reference_set 不能为空")
        self._reference = reference_set

    async def run(self, reviewer_llm) -> CalibrationReport:
        """用当前 LLM 重新评审校准集，与基准分对比。

        Args:
            reviewer_llm: 当前使用的评审 LLM 实例

        Returns:
            CalibrationReport — 含平均偏离、最大偏离、告警信号
        """

        deviations: list[float] = []
        details: list[tuple[int, float, float]] = []

        for sample in self._reference:
            current = await self._evaluate(sample, reviewer_llm)
            deviation = current - sample.baseline_score
            deviations.append(deviation)
            details.append((sample.chapter_index, sample.baseline_score, current))

        avg_dev = sum(deviations) / len(deviations)
        max_dev = max(abs(d) for d in deviations)

        return CalibrationReport(
            avg_deviation=avg_dev,
            max_deviation=max_dev,
            sample_count=len(self._reference),
            alarm=abs(avg_dev) > 5.0,  # 平均偏离 >5 分告警
            details=details,
        )

    async def _evaluate(self, sample: CalibrationSample, reviewer_llm) -> float:
        """对单个样本做快速评分（仅四维评分，跳过完整辩论）。"""
        prompt = (
            f"请对以下章节进行四维评分，仅输出 JSON：\n"
            f'{{\n  "quality_score": <0-100>\n}}\n'
            f"不要输出解释。\n\n"
            f"## 章节内容\n{sample.chapter_text[:3000]}"
        )
        try:
            from novelfactory.agents.infra import async_llm_call_with_retry

            response = await async_llm_call_with_retry(
                reviewer_llm, prompt, step_name="calibration_eval"
            )
            raw = response.content if hasattr(response, "content") else str(response)
            import json as _json

            result = _json.loads(raw)
            return float(result.get("quality_score", sample.baseline_score))
        except Exception:
            return sample.baseline_score  # 失败时返回基准分（无变化）
