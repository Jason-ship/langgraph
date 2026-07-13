"""AI味传感器 — 封装 analysis/ai_style_analyzer.py 为 ProgrammaticReport 产出。

适配器模式：不修改原始分析器，在新模块中封装为统一产出格式。
原始分析器保持不变，其他模块仍可直接调用。

职责：
    调用 analyze_ai_style() → 转换为 AIStyleMetricsBrief + issues + fix
"""

from __future__ import annotations

import logging
from typing import Any

from novelfactory.analysis.ai_style_analyzer import analyze_ai_style
from novelfactory.evaluation.schemas import AIStyleMetricsBrief

logger = logging.getLogger(__name__)


class AIStyleSensor:
    """AI味传感器 — 纯代码，零 LLM。

    调用现有的 analyze_ai_style() 函数，将结果转换为统一的 AIStyleMetricsBrief。
    """

    def analyze(
        self,
        chapter_text: str,
        genre: str | None = None,
    ) -> tuple[float, AIStyleMetricsBrief, list[str], str, bool]:
        """执行 AI 味检测。

        Args:
            chapter_text: 章节文本
            genre: 题材名称（用于题材感知豁免）

        Returns:
            (ai_style_score, metrics_brief, issues, fix_suggestion, is_short_text)
            - ai_style_score: 0-1，越低越好
            - metrics_brief: 8 维指标摘要
            - issues: 具体问题列表
            - fix_suggestion: 修改建议
            - is_short_text: 是否短文本
        """
        result: dict[str, Any] = analyze_ai_style(chapter_text, genre=genre)

        ai_style_score: float = result["ai_style_score"]
        is_short_text: bool = result.get("details", {}).get("is_short_text", False)

        # 转换 TypedDict metrics 为 Pydantic model
        raw_metrics = result["metrics"]
        metrics_brief = AIStyleMetricsBrief(
            repetition_ngram=raw_metrics.get("repetition_ngram", 0.0),
            sentence_length_variance=raw_metrics.get("sentence_length_variance", 0.0),
            lexical_diversity=raw_metrics.get("lexical_diversity", 0.0),
            cliche_ratio=raw_metrics.get("cliche_ratio", 0.0),
            punctuation_rhythm=raw_metrics.get("punctuation_rhythm", 0.0),
            dialogue_ratio=raw_metrics.get("dialogue_ratio", 0.0),
            sensory_emotion_density=raw_metrics.get("sensory_emotion_density", 0.0),
            semantic_smoothness=raw_metrics.get("semantic_smoothness", 0.0),
        )

        issues: list[str] = result.get("issues", [])

        # 生成修改建议
        fix_suggestion = self._build_fix_suggestion(issues, ai_style_score)

        logger.info(
            "[AI味传感器] score=%.4f short=%s issues=%d",
            ai_style_score,
            is_short_text,
            len(issues),
        )

        return ai_style_score, metrics_brief, issues, fix_suggestion, is_short_text

    def _build_fix_suggestion(
        self,
        issues: list[str],
        ai_style_score: float,
    ) -> str:
        """根据检测结果生成修改建议。"""
        if not issues or ai_style_score <= 0.3:
            return ""

        parts: list[str] = []
        for issue in issues:
            # 提取方括号中的类型标签
            if "【" in issue and "】" in issue:
                label = issue[issue.index("【") + 1 : issue.index("】")]
                parts.append(f"- {label}: {issue[issue.index('】') + 1 :]}")
            else:
                parts.append(f"- {issue}")

        return "AI味修改建议：\n" + "\n".join(parts)
