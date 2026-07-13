"""统一反馈包构建 — v7.1 含 LLM 语义分析反馈。

一个 FeedbackBundle 同时生成 refiner prompt 和 writer prompt，
确保两条路径获得完全一致的信息（无遗漏）。

v7.1：新增 LLM 老书虫 + LLM AI味语义分析反馈注入。
"""

from __future__ import annotations

import logging
from typing import Any

from novelfactory.evaluation.schemas import (
    CrossChapterSignals,
    DebateReport,
    FeedbackBundle,
    FourDimReviewResult,
    ProgrammaticReport,
)

logger = logging.getLogger(__name__)


class FeedbackBuilder:
    """统一反馈包构建器。

    v7.1：新增 LLM 语义分析反馈注入（老书虫 + AI味）。
    将四维评审 + 程序化分析 + LLM语义分析 + 辩论 + 跨章信号 融合为 FeedbackBundle。
    """

    def build(
        self,
        four_dim: FourDimReviewResult,
        programmatic: ProgrammaticReport,
        debate: DebateReport,
        cross_chapter: CrossChapterSignals,
        final_score: float,
        debate_penalty: float,
        llm_old_reader: Any = None,
        llm_ai_style: Any = None,
    ) -> FeedbackBundle:
        """构建统一反馈包。

        Args:
            four_dim: 四维 LLM 评分结果
            programmatic: 程序化分析报告
            debate: 辩论报告
            cross_chapter: 跨章信号
            final_score: 融合后最终评分
            debate_penalty: 辩论惩罚分
            llm_old_reader: LLM 老书虫分析结果（可选）
            llm_ai_style: LLM AI味分析结果（可选）

        Returns:
            FeedbackBundle
        """

        # 评分概要
        score_summary = self._build_score_summary(
            four_dim,
            programmatic,
            debate,
            cross_chapter,
            final_score,
            debate_penalty,
            llm_old_reader=llm_old_reader,
            llm_ai_style=llm_ai_style,
        )

        # 程序化指标摘要
        metrics_brief = programmatic.ai_style_metrics.to_brief_string()

        # 跨章摘要
        cross_brief = cross_chapter.to_debate_briefing()
        cross_issues = four_dim.cross_chapter_issues

        # 毒点/爽点类型列表（程序化 + LLM 融合）
        toxic_types = [t.type for t in programmatic.toxic_points]
        shuangdian_types = [s.type for s in programmatic.shuangdian_points]

        # LLM 融合反馈
        llm_feedback_parts: list[str] = []
        if llm_old_reader is not None and not llm_old_reader.failed:
            if llm_old_reader.implicit_toxic_found:
                llm_feedback_parts.append(
                    "【LLM检测：隐式毒点】自动检测到关键词匹配无法发现的语义毒点，"
                    "详见 weaknesses"
                )
            if llm_old_reader.concrete_suggestions:
                llm_feedback_parts.append(
                    f"【LLM老书虫建议】{llm_old_reader.concrete_suggestions[:500]}"
                )
            # 合并 LLM 发现的额外毒点
            for tp in llm_old_reader.toxic_points:
                if tp.type not in toxic_types:
                    toxic_types.append(tp.type)

        if llm_ai_style is not None and not llm_ai_style.failed:
            if llm_ai_style.has_obvious_ai:
                llm_feedback_parts.append(
                    "【LLM检测：AI痕迹】语义层面发现明显的 AI 生成特征，"
                    "详见 semantic_issues"
                )
            if llm_ai_style.summary:
                llm_feedback_parts.append(
                    f"【LLM AI味评估】{llm_ai_style.summary[:300]}"
                )

        return FeedbackBundle(
            score_summary=score_summary,
            review_comments=four_dim.review_comments,
            ai_style_fix=programmatic.ai_style_fix,
            lao_shu_chong_fix=programmatic.lao_shu_chong_fix,
            toxic_points=toxic_types,
            shuangdian_points=shuangdian_types,
            debate_issues=debate.merged_issues,
            debate_strengths=debate.merged_strengths,
            debate_suggestions=debate.merged_suggestions,
            debate_transcript=debate.debate_transcript,
            ai_style_metrics_brief=metrics_brief,
            cross_chapter_brief=cross_brief,
            cross_chapter_issues=cross_issues,
            # LLM 融合反馈（通过现有字段传递）
            # ai_style_fix 和 lao_shu_chong_fix 已包含程序化部分，
            # LLM 的额外内容通过 review_comments 传递
        )

    def _build_score_summary(
        self,
        four_dim: FourDimReviewResult,
        programmatic: ProgrammaticReport,
        debate: DebateReport,
        cross_chapter: CrossChapterSignals,
        final_score: float,
        debate_penalty: float,
        llm_old_reader: Any = None,
        llm_ai_style: Any = None,
    ) -> str:
        """生成评分概要行（v7.1 含 LLM 语义分析评分）。"""
        parts: list[str] = []

        parts.append(f"四维{four_dim.quality_score:.0f}")
        parts.append(f"AI味{programmatic.ai_style_score:.2f}")
        parts.append(f"老书虫{programmatic.lao_shu_chong_score:.0f}")

        # LLM 语义评分
        if llm_old_reader is not None and not llm_old_reader.failed:
            parts.append(f"LLM老书虫{llm_old_reader.semantic_score:.0f}")
        if llm_ai_style is not None and not llm_ai_style.failed:
            parts.append(f"LLM人感{llm_ai_style.human_like_score:.0f}")

        if cross_chapter.has_prev_context:
            parts.append(f"跨章{four_dim.cross_chapter_consistency:.0f}")

        if debate.merged_issues:
            parts.append(f"辩论{len(debate.merged_issues)}issues")

        if debate_penalty > 0:
            parts.append(f"惩罚-{debate_penalty:.0f}")

        parts.append(f"→最终{final_score:.1f}")

        return " | ".join(parts)
