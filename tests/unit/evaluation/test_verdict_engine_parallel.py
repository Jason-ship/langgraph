"""VerdictEngine.evaluate() 统一评审集成验收测试（v8.2）。

v8.2 重构后 evaluate() 走 UnifiedReviewEngine 单次调用五视角评审，
替代旧"5 个 LLM 维度 + 程序化传感器"并行结构。

测试方式：monkeypatch UnifiedReviewEngine.evaluate 为 async 桩，
验证：调用一次、结果融合正确、severe 毒点强制重写、次数用尽兜底。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from novelfactory.evaluation.schemas import AttemptInfo
from novelfactory.evaluation.unified import UnifiedFourDim, UnifiedReviewResult
from novelfactory.evaluation.verdict.engine import VerdictEngine

CHAPTER_TEXT = "测试内容。" * 200
GENRE = "都市"
GENRE_GUIDE = ""
PREV_SUMMARY = ""
CHAPTER_INDEX = 1
ATTEMPT_INFO = AttemptInfo(loop_count=0, refine_attempts=0, max_rewrite=5, max_refine=2)


def _mock_llm() -> SimpleNamespace:
    return SimpleNamespace(model="mock-llm", invoke=lambda *a, **k: None)


def _ok_review() -> UnifiedReviewResult:
    """合格章节评审结果（无 severe 毒点、3 个爽点）。"""
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


def _severe_review() -> UnifiedReviewResult:
    """含 severe 毒点的评审结果（触发强制重写）。"""
    r = _ok_review()
    r.final_score = 78.0
    r.toxic_points = [{"type": "NTR", "paragraph": 8, "severity": "severe", "reason": "x"}]
    r.severe_toxic = True
    return r


def _failed_review() -> UnifiedReviewResult:
    return UnifiedReviewResult(failed=True, final_score=60.0)


async def _run_evaluate(monkeypatch: pytest.MonkeyPatch, review: UnifiedReviewResult, attempt: AttemptInfo | None = None):
    """执行一次 evaluate，将 UnifiedReviewEngine.evaluate 替换为桩。"""
    calls: list[dict] = []

    async def stub_evaluate(self, **kwargs):
        calls.append(kwargs)
        return review

    monkeypatch.setattr("novelfactory.evaluation.unified.engine.UnifiedReviewEngine.evaluate", stub_evaluate)

    engine = VerdictEngine()
    verdict = await engine.evaluate(
        chapter_text=CHAPTER_TEXT,
        genre=GENRE,
        genre_scoring_guide=GENRE_GUIDE,
        prev_summary=PREV_SUMMARY,
        chapter_index=CHAPTER_INDEX,
        attempt_info=attempt or ATTEMPT_INFO,
        reviewer_llm=_mock_llm(),
        debate_llm=_mock_llm(),
    )
    return verdict, calls


class TestUnifiedIntegration:
    """统一评审集成。"""

    async def test_evaluate_calls_unified_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """evaluate 应恰好调用 UnifiedReviewEngine 1 次（替代 5 维度并行）。"""
        verdict, calls = await _run_evaluate(monkeypatch, _ok_review())
        assert len(calls) == 1
        assert calls[0]["genre"] == GENRE
        assert calls[0]["guide"] == GENRE_GUIDE
        assert verdict is not None

    async def test_evaluate_merges_unified_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """统一评审结果应正确融合进 VerdictResult。"""
        verdict, _ = await _run_evaluate(monkeypatch, _ok_review())

        assert verdict.final_score == pytest.approx(82.0)
        assert verdict.quality_score == pytest.approx(88.0)  # 四维总分
        assert verdict.programmatic_score == 0.0  # 程序化已移除，字段保留兼容
        assert verdict.cross_chapter_consistency == pytest.approx(84.0)
        assert verdict.ai_style_score == pytest.approx(0.75)
        assert verdict.lao_shu_chong_score == pytest.approx(82.0)
        assert verdict.has_severe_toxic is False
        assert verdict.feedback.toxic_points == []

    async def test_severe_toxic_forces_rewrite(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM 判定 severe 毒点 → 未耗尽重写时强制 REWRITE。"""
        verdict, _ = await _run_evaluate(monkeypatch, _severe_review())
        assert verdict.level.value == "rewrite"
        assert verdict.has_severe_toxic is True

    async def test_failed_review_degrade_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """统一评审失败（failed）→ 降级 PASS（v8.5-fix：评审失败≠质量差，避免 API 故障触发整章重写）。"""
        verdict, _ = await _run_evaluate(monkeypatch, _failed_review())
        assert verdict.level.value == "pass"

    async def test_exhausted_force_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """双向次数用尽 → 强制 PASS（防死循环兜底）。"""
        exhausted = AttemptInfo(loop_count=5, refine_attempts=2, max_rewrite=5, max_refine=2)
        verdict, _ = await _run_evaluate(monkeypatch, _severe_review(), attempt=exhausted)
        assert verdict.level.value == "pass"

    async def test_iteration_bonus_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """迭代宽松加分保留：重写/润色次数越多分数越高（封顶）。"""
        attempt = AttemptInfo(loop_count=2, refine_attempts=1, max_rewrite=5, max_refine=2)
        verdict, _ = await _run_evaluate(monkeypatch, _ok_review(), attempt=attempt)
        # v9.1: 82 + min(2*2 + 1*1, 4) = 86（加分收紧：rewrite=2/refine=1/封顶4）
        assert verdict.final_score == pytest.approx(86.0)
