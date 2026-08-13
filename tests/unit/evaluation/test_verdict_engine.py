"""VerdictEngine 单元测试 — 统一评审融合逻辑（v8.2）。

覆盖:
    - _fuse_unified() 统一评审融合（LLM 综合分 + 迭代加分）
    - _decide_unified_level() 三级决议（PASS/REFINE/REWRITE + severe 毒点强制重写 + 次数兜底）
    - _build_unified_feedback() 反馈包构建
"""

from __future__ import annotations

import pytest

from novelfactory.evaluation.schemas import AttemptInfo, VerdictLevel
from novelfactory.evaluation.unified import UnifiedFourDim, UnifiedReviewResult
from novelfactory.evaluation.verdict.engine import VerdictEngine


@pytest.fixture
def engine() -> VerdictEngine:
    return VerdictEngine()


def _ok_review() -> UnifiedReviewResult:
    return UnifiedReviewResult(
        final_score=82.0,
        four_dim=UnifiedFourDim(logic=27.0, writing=21.0, character=22.0, world=18.0),
        shuangdian_points=["打脸|P5", "升级|P14", "奖励|P20"],
        shuangdian_count=3,
        human_like_score=0.75,
        attraction_score=88.0,
        immersion_score=90.0,
        cross_chapter_score=84.0,
        review_comments="整体合格",
        fix_suggestions=["P5 补一句反派反应"],
    )


def _attempt(loop: int = 0, refine: int = 0) -> AttemptInfo:
    return AttemptInfo(loop_count=loop, refine_attempts=refine, max_rewrite=5, max_refine=2)


class TestFuseUnified:
    """统一评审融合。"""

    def test_merges_fields(self, engine: VerdictEngine) -> None:
        verdict = engine._fuse_unified(_ok_review(), _attempt(), chapter_length=4000)
        assert verdict.final_score == pytest.approx(82.0)
        assert verdict.quality_score == pytest.approx(88.0)
        assert verdict.programmatic_score == 0.0
        assert verdict.cross_chapter_consistency == pytest.approx(84.0)
        assert verdict.ai_style_score == pytest.approx(0.75)
        assert verdict.lao_shu_chong_score == pytest.approx(82.0)
        assert verdict.has_severe_toxic is False
        assert verdict.feedback.score_summary
        assert verdict.feedback.shuangdian_points == _ok_review().shuangdian_points

    def test_iteration_bonus(self, engine: VerdictEngine) -> None:
        verdict = engine._fuse_unified(_ok_review(), _attempt(loop=2, refine=1), chapter_length=4000)
        # v9.1: 82 + min(2*2 + 1*1, 4) = 86（加分收紧：rewrite=2/refine=1/封顶4）
        assert verdict.final_score == pytest.approx(86.0)

    def test_iteration_bonus_capped(self, engine: VerdictEngine) -> None:
        verdict = engine._fuse_unified(_ok_review(), _attempt(loop=5, refine=2), chapter_length=4000)
        assert verdict.final_score == pytest.approx(86.0)  # 封顶 4 分


class TestDecideUnifiedLevel:
    """统一评审路由三态。"""

    def test_pass_high_score(self, engine: VerdictEngine) -> None:
        r = _ok_review()
        level = engine._decide_unified_level(82.0, r, _attempt())
        assert level == VerdictLevel.PASS

    def test_passed_flag_matches_level(self, engine: VerdictEngine) -> None:
        """passed 字段应与级别一致（PASS=True，其余 False）。"""
        verdict = engine._fuse_unified(_ok_review(), _attempt(), chapter_length=4000)
        assert verdict.level == VerdictLevel.PASS
        assert verdict.passed is True
        r = _ok_review()
        r.final_score = 45.0
        verdict_low = engine._fuse_unified(r, _attempt(), chapter_length=4000)
        assert verdict_low.level == VerdictLevel.REWRITE
        assert verdict_low.passed is False

    def test_refine_mid_score(self, engine: VerdictEngine) -> None:
        r = _ok_review()
        r.final_score = 60.0
        level = engine._decide_unified_level(60.0, r, _attempt())
        assert level == VerdictLevel.REFINE

    def test_rewrite_low_score(self, engine: VerdictEngine) -> None:
        r = _ok_review()
        r.final_score = 45.0
        level = engine._decide_unified_level(45.0, r, _attempt())
        assert level == VerdictLevel.REWRITE

    def test_severe_toxic_forces_rewrite(self, engine: VerdictEngine) -> None:
        r = _ok_review()
        r.severe_toxic = True
        level = engine._decide_unified_level(80.0, r, _attempt())
        assert level == VerdictLevel.REWRITE

    def test_failed_review_rewrites(self, engine: VerdictEngine) -> None:
        r = UnifiedReviewResult(failed=True, final_score=60.0)
        level = engine._decide_unified_level(60.0, r, _attempt())
        assert level == VerdictLevel.REWRITE

    def test_exhausted_force_pass(self, engine: VerdictEngine) -> None:
        r = _ok_review()
        r.severe_toxic = True
        level = engine._decide_unified_level(50.0, r, _attempt(loop=5, refine=2))
        assert level == VerdictLevel.PASS


class TestBuildUnifiedFeedback:
    """统一反馈包。"""

    def test_feedback_contains_unified_summary(self, engine: VerdictEngine) -> None:
        fb = engine._build_unified_feedback(_ok_review())
        assert "统一评审" in fb.score_summary
        assert "82" in fb.score_summary
        assert fb.toxic_points == []
        assert fb.debate_issues == []
        assert fb.debate_suggestions == "P5 补一句反派反应"

    def test_feedback_contains_toxic(self, engine: VerdictEngine) -> None:
        r = _ok_review()
        r.toxic_points = [{"type": "POWER_BREAK", "paragraph": 8, "severity": "severe", "reason": "x"}]
        fb = engine._build_unified_feedback(r)
        assert fb.toxic_points == ["POWER_BREAK"]
