"""InformedDebateEngine 单元测试。

覆盖:
    - severity_weight 计算
    - 收敛判定
    - 辩论失败降级
"""

from __future__ import annotations

import pytest

from novelfactory.evaluation.schemas import DebateReport, PerspectiveReview, Rebuttal


class TestDebateSeverityWeight:
    """DebateReport.severity_weight 计算测试。"""

    def test_no_issues_zero_penalty(self):
        """无问题 → 惩罚为 0。"""
        report = DebateReport(
            debate_failed=False,
            convergence_achieved=True,
            merged_issues=[],
            merged_strengths=[],
            merged_suggestions="",
            debate_transcript="",
        )
        assert report.severity_weight == 0.0

    def test_few_issues_small_penalty(self):
        """少量问题 → 小惩罚。"""
        report = DebateReport(
            debate_failed=False,
            convergence_achieved=True,
            merged_issues=["对话略显生硬", "结尾仓促"],
            merged_strengths=[],
            merged_suggestions="",
            debate_transcript="",
        )
        assert report.severity_weight <= 10.0

    def test_many_issues_capped(self):
        """大量问题 → 不超上限。"""
        from novelfactory.config.constants import VERDICT_DEBATE_PENALTY_CAP

        issues = [f"问题{i}" for i in range(20)]
        report = DebateReport(
            debate_failed=False,
            convergence_achieved=True,
            merged_issues=issues,
            merged_strengths=[],
            merged_suggestions="",
            debate_transcript="",
        )
        assert report.severity_weight <= VERDICT_DEBATE_PENALTY_CAP

    def test_converged_vs_not_converged(self):
        """收敛时惩罚可信度高，未收敛时折半（需足够多问题以展示 CAP 差异）。"""
        issues = [f"问题{i}" for i in range(15)]  # 足够多问题使未收敛撞到折半 CAP(9.0)
        converged = DebateReport(
            debate_failed=False,
            convergence_achieved=True,
            merged_issues=issues,
            merged_strengths=[],
            merged_suggestions="",
            debate_transcript="",
        )
        not_converged = DebateReport(
            debate_failed=False,
            convergence_achieved=False,
            merged_issues=issues,
            merged_strengths=[],
            merged_suggestions="",
            debate_transcript="",
        )
        assert not_converged.severity_weight <= converged.severity_weight

    def test_severe_issues_extra_penalty(self):
        """严重问题加重惩罚。"""
        report = DebateReport(
            debate_failed=False,
            convergence_achieved=True,
            merged_issues=["严重毒点: 逻辑断裂"],
            merged_strengths=[],
            merged_suggestions="",
            debate_transcript="",
        )
        assert report.severity_weight > 0.0

    def test_debate_failed_zero_penalty(self):
        """辩论失败 → 形态上 weight 仍可计算但不用于评分（调用方负责降级）。"""
        report = DebateReport(
            debate_failed=True,
            convergence_achieved=False,
            merged_issues=[],
            merged_strengths=[],
            merged_suggestions="",
            debate_transcript="辩论失败: 超时",
        )
        # debate_failed 不影响 weight 计算逻辑
        assert report.severity_weight >= 0.0


class TestDebateConvergence:
    """辩论收敛判定测试。"""

    def test_converged_metadata(self):
        """提前收敛标记不影响其他字段。"""
        report = DebateReport(
            debate_failed=False,
            convergence_achieved=True,
            merged_issues=["小问题"],
            merged_strengths=["主线清晰"],
            merged_suggestions="调整对话节奏",
            debate_transcript="编辑: ... 读者: ...",
        )
        assert report.convergence_achieved is True
        assert len(report.merged_issues) == 1
        assert len(report.merged_strengths) == 1

    def test_debate_failed_metadata(self):
        """辩论失败降级标记。"""
        report = DebateReport(
            debate_failed=True,
            convergence_achieved=False,
            merged_issues=[],
            merged_strengths=[],
            merged_suggestions="",
            debate_transcript="辩论失败: LLM 返回空",
        )
        assert report.debate_failed is True
        assert report.debate_transcript != ""
