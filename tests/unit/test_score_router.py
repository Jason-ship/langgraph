"""P0: verdict_router (v6.3) — 3 路路由替代旧版 _score_router。

verdict_router 替代了 _score_router 的 12 分支，改为基于 VerdictLevel 的 3 路纯路由。
所有兜底逻辑（次数用尽、短文本、scorer 故障）已在 VerdictEngine 中处理。
"""

from __future__ import annotations

import pytest


class FakeWritingState:
    """Fake state for verdict_router tests."""
    def __init__(self, level: str = "rewrite"):
        self.verdict_result = {"level": level}

    def get(self, key, default=0):
        return getattr(self, key, default)


@pytest.fixture(scope="module")
def router():
    from novelfactory.evaluation.verdict.router import verdict_router
    return verdict_router


# ── Branch 1: PASS → __exit_for_chapter__ ─────────────────────────────────

class TestPassExit:
    def test_level_pass_exits(self, router):
        state = FakeWritingState(level="pass")
        assert router(state) == "__exit_for_chapter__"

    def test_pass_with_extra_data_exits(self, router):
        state = FakeWritingState(level="pass")
        state.verdict_result["final_score"] = 88.5
        assert router(state) == "__exit_for_chapter__"


# ── Branch 2: REWRITE → chapter_planner ──────────────────────────────────

class TestRewriteBranch:
    def test_level_rewrite_goes_to_planner(self, router):
        state = FakeWritingState(level="rewrite")
        assert router(state) == "chapter_planner"

    def test_missing_verdict_defaults_to_rewrite(self, router):
        """Missing verdict_result defaults to rewrite."""
        state = FakeWritingState(level="rewrite")
        state.verdict_result = {}
        assert router(state) == "chapter_planner"


# ── Branch 3: REFINE → chapter_refiner ───────────────────────────────────

class TestRefineBranch:
    def test_level_refine_goes_to_refiner(self, router):
        state = FakeWritingState(level="refine")
        assert router(state) == "chapter_refiner"

# ── Branch 4: VerdictLevel enum support ──────────────────────────────────

class TestVerdictLevelEnum:
    def test_verdict_level_enum_pass(self, router):
        from novelfactory.evaluation.schemas import VerdictLevel
        state = FakeWritingState(level=VerdictLevel.PASS.value)
        assert router(state) == "__exit_for_chapter__"

    def test_verdict_level_enum_refine(self, router):
        from novelfactory.evaluation.schemas import VerdictLevel
        state = FakeWritingState(level=VerdictLevel.REFINE.value)
        assert router(state) == "chapter_refiner"

    def test_verdict_level_enum_rewrite(self, router):
        from novelfactory.evaluation.schemas import VerdictLevel
        state = FakeWritingState(level=VerdictLevel.REWRITE.value)
        assert router(state) == "chapter_planner"


# ── Branch 5: Edge cases ─────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_state_defaults_to_rewrite(self):
        """Completely empty state defaults to rewrite."""
        from novelfactory.evaluation.verdict.router import verdict_router
        assert verdict_router({}) == "chapter_planner"

    def test_unknown_level_defaults_to_rewrite(self, router):
        """Unknown verdict level falls back to rewrite."""
        state = FakeWritingState(level="UNKNOWN_LEVEL")
        assert router(state) == "chapter_planner"
