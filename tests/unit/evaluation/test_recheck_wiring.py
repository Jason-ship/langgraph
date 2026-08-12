"""coordinator 轻量复查接线测试（v8.3）。

覆盖:
    - _build_recheck_issues() 问题清单提取
    - _try_quick_recheck() 触发条件 / 采信 / 兜底
"""

from __future__ import annotations

import asyncio

import pytest

from novelfactory.evaluation.coordinator import _build_recheck_issues, _try_quick_recheck
from novelfactory.evaluation.schemas import (
    AttemptInfo,
    FeedbackBundle,
    VerdictLevel,
    VerdictResult,
)

RECHECK_OK = """<review_analysis>P8 战力铺垫已补，问题清除。</review_analysis>
[复查] passed=true
[评分] final=80.0
[毒点] 无
[新增问题] 无
[分数变化] final: 68.0 -> 80.0"""

RECHECK_LOW = """<review_analysis>P8 仍未修复。</review_analysis>
[复查] passed=false
[评分] final=50.0
[毒点] POWER_BREAK|P8
[新增问题] 无
[分数变化] final: 68.0 -> 50.0"""


class _FakeLlm:
    def __init__(self, payload: str):
        self._payload = payload

    async def ainvoke(self, prompt: str):
        class _R:
            content = self._payload
        return _R()


class TestBuildRecheckIssues:
    def test_extracts_all_issue_sources(self):
        prev = {
            "level": "REFINE",
            "final_score": 68.0,
            "feedback": {
                "toxic_points": ["POWER_BREAK"],
                "debate_issues": ["人物动机不足"],
                "debate_suggestions": "P8 补铺垫\nP12 删水文",
            },
        }
        issues = _build_recheck_issues(prev)
        assert "POWER_BREAK" in issues
        assert "人物动机不足" in issues
        assert "P8 补铺垫" in issues
        assert "P12 删水文" in issues

    def test_empty_when_no_feedback(self):
        assert _build_recheck_issues({}) == []


class TestTryQuickRecheck:
    def test_skips_first_round(self):
        r = asyncio.run(
            _try_quick_recheck("正文", 0, 0, {"level": "REFINE"}, _FakeLlm(RECHECK_OK))
        )
        assert r is None

    def test_skips_when_prev_level_is_pass(self):
        r = asyncio.run(
            _try_quick_recheck("正文", 1, 0, {"level": "PASS"}, _FakeLlm(RECHECK_OK))
        )
        assert r is None

    def test_accepts_when_score_met(self):
        prev = {
            "level": "REFINE",
            "final_score": 68.0,
            "feedback": {"toxic_points": ["POWER_BREAK"], "debate_suggestions": "P8 补铺垫"},
        }
        r = asyncio.run(
            _try_quick_recheck("正文", 0, 1, prev, _FakeLlm(RECHECK_OK))
        )
        assert r is not None
        assert r.failed is False
        assert r.final_score == 80.0

    def test_falls_back_when_score_low(self):
        prev = {
            "level": "REWRITE",
            "final_score": 50.0,
            "feedback": {"toxic_points": ["POWER_BREAK"]},
        }
        r = asyncio.run(
            _try_quick_recheck("正文", 1, 0, prev, _FakeLlm(RECHECK_LOW))
        )
        assert r is None  # 复查未达标 → 完整评审兜底

    def test_falls_back_when_recheck_failed(self):
        prev = {"level": "REFINE", "final_score": 68.0, "feedback": {"toxic_points": ["x"]}}
        r = asyncio.run(
            _try_quick_recheck("正文", 1, 0, prev, _FakeLlm("无法解析"))
        )
        assert r is None

    def test_skips_when_no_issues(self):
        r = asyncio.run(
            _try_quick_recheck("正文", 1, 0, {"level": "REFINE", "feedback": {}}, _FakeLlm(RECHECK_OK))
        )
        assert r is None


def _real_prev_verdict() -> dict:
    """构造与真实 state 一致的 verdict_result（VerdictResult.model_dump）。"""
    return VerdictResult(
        level=VerdictLevel.REFINE,
        passed=False,
        final_score=72.0,
        quality_score=74.0,
        programmatic_score=0.0,
        cross_chapter_consistency=85.0,
        debate_penalty=0.0,
        ai_style_score=0.10,
        lao_shu_chong_score=72.0,
        feedback=FeedbackBundle(
            score_summary="统一评审：72/100",
            review_comments="需润色",
            toxic_points=["POWER_BREAK"],
            shuangdian_points=[],
            debate_issues=["战力突兀"],
            debate_strengths=[],
            debate_suggestions="P8 补铺垫",
        ),
        attempt_info=AttemptInfo(loop_count=0, refine_attempts=1, max_rewrite=5, max_refine=2),
        has_severe_toxic=False,
    ).model_dump()


class TestRealModelDumpStructure:
    """回归防护：真实 state 中 level 是 VerdictLevel 枚举对象（model_dump 保留枚举）。"""

    def test_issues_extracted_from_real_structure(self):
        issues = _build_recheck_issues(_real_prev_verdict())
        assert "POWER_BREAK" in issues
        assert "战力突兀" in issues
        assert "P8 补铺垫" in issues

    def test_level_enum_accepts_refine(self):
        r = asyncio.run(
            _try_quick_recheck("正文", 0, 1, _real_prev_verdict(), _FakeLlm(RECHECK_OK))
        )
        assert r is not None
        assert r.failed is False
        assert r.final_score == 80.0
