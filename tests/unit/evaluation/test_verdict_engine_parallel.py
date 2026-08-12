"""VerdictEngine.evaluate() asyncio.gather 并行化验收测试。

验证 v7.9 并行化重构：5 个 LLM 评审维度
（llm_old_reader / llm_ai_style / llm_attraction / debate / four_dim）
通过 asyncio.gather 并行执行而非串行。

测试方式：monkeypatch 将 5 个维度替换为 async 桩函数，
用耗时上界断言并行性；用固定返回值断言结果融合正确性。

注意：
    - evaluate() 中 run_programmatic_analysis 为真实纯计算代码，
      提供足够的 chapter_text 保证程序化分析正常执行。
    - 桩函数不访问真实 LLM，reviewer_llm / debate_llm 传简单对象即可。
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from novelfactory.evaluation.llm.schemas import (
    LLMAIStyleResult,
    LLMAttractionResult,
    LLMOldReaderResult,
)
from novelfactory.evaluation.schemas import (
    AttemptInfo,
    DebateReport,
    FourDimReviewResult,
    VerdictResult,
)
from novelfactory.evaluation.verdict.engine import VerdictEngine

# ═══════════════════════════════════════════════════════════════════════════════
#  公共常量
# ═══════════════════════════════════════════════════════════════════════════════

CHAPTER_TEXT = "测试内容。" * 200  # 足够长的章节文本，保证程序化分析可执行
GENRE = "都市"
GENRE_GUIDE = ""
PREV_SUMMARY = ""
CHAPTER_INDEX = 1
ATTEMPT_INFO = AttemptInfo(loop_count=0, refine_attempts=0, max_rewrite=5, max_refine=2)

# 桩返回值（使用真实 Schema 对象，保证 _fuse 融合链完整可运行）
STUB_OLD_READER = LLMOldReaderResult(semantic_score=82.0)
STUB_AI_STYLE = LLMAIStyleResult(human_like_score=78.0)
STUB_ATTRACTION = LLMAttractionResult(
    attraction_score=74.0, fix="吸引力专家建议：增强开篇钩子"
)
STUB_DEBATE = DebateReport(convergence_achieved=True, merged_strengths=["文笔流畅"])
STUB_FOUR_DIM = FourDimReviewResult(
    quality_score=80.0,
    review_comments="整体合格",
    cross_chapter_consistency=75.0,
)


def _mock_llm() -> SimpleNamespace:
    """桩 LLM 对象（五个维度桩函数不使用真实 LLM）。"""
    return SimpleNamespace(model="mock-llm", invoke=lambda *a, **k: None)


def _patch_five_dimensions(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """将 evaluate() 依赖的 5 个 LLM 维度替换为桩函数。

    Returns:
        dict 映射维度名 → 桩函数（供调用方记录/断言）
    """
    engine_mod = "novelfactory.evaluation.verdict.engine"

    async def stub_old_reader(**kwargs: object) -> LLMOldReaderResult:
        return STUB_OLD_READER

    async def stub_ai_style(**kwargs: object) -> LLMAIStyleResult:
        return STUB_AI_STYLE

    async def stub_attraction(**kwargs: object) -> LLMAttractionResult:
        return STUB_ATTRACTION

    async def stub_debate(self: object, *args: object, **kwargs: object) -> DebateReport:
        return STUB_DEBATE

    async def stub_four_dim(
        self: object, *args: object, **kwargs: object
    ) -> FourDimReviewResult:
        return STUB_FOUR_DIM

    monkeypatch.setattr(f"{engine_mod}.llm_old_reader_analysis", stub_old_reader)
    monkeypatch.setattr(f"{engine_mod}.llm_ai_style_analysis", stub_ai_style)
    monkeypatch.setattr(f"{engine_mod}.attraction_llm_analysis", stub_attraction)
    monkeypatch.setattr(VerdictEngine, "_run_debate", stub_debate)
    monkeypatch.setattr(VerdictEngine, "_run_four_dim_review", stub_four_dim)

    return {
        "old_reader": stub_old_reader,
        "ai_style": stub_ai_style,
        "attraction": stub_attraction,
        "debate": stub_debate,
        "four_dim": stub_four_dim,
    }


async def _run_evaluate() -> VerdictResult:
    """以固定输入执行一次完整 evaluate()。"""
    engine = VerdictEngine()
    return await engine.evaluate(
        chapter_text=CHAPTER_TEXT,
        genre=GENRE,
        genre_scoring_guide=GENRE_GUIDE,
        prev_summary=PREV_SUMMARY,
        chapter_index=CHAPTER_INDEX,
        attempt_info=ATTEMPT_INFO,
        reviewer_llm=_mock_llm(),
        debate_llm=_mock_llm(),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  并行性验证
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluateParallel:
    """asyncio.gather 并行化验证。"""

    async def test_evaluate_parallel_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """5 个维度应并行执行：总耗时显著小于串行耗时（0.3s × 5 = 1.5s）。"""
        engine_mod = "novelfactory.evaluation.verdict.engine"
        sleep_duration = 0.3
        active = 0
        max_active = 0
        lock = asyncio.Lock()

        async def stub_old_reader(**kwargs: object) -> LLMOldReaderResult:
            nonlocal active, max_active
            async with lock:
                active += 1
                max_active = max(max_active, active)
            await asyncio.sleep(sleep_duration)
            async with lock:
                active -= 1
            return STUB_OLD_READER

        async def stub_ai_style(**kwargs: object) -> LLMAIStyleResult:
            nonlocal active, max_active
            async with lock:
                active += 1
                max_active = max(max_active, active)
            await asyncio.sleep(sleep_duration)
            async with lock:
                active -= 1
            return STUB_AI_STYLE

        async def stub_attraction(**kwargs: object) -> LLMAttractionResult:
            nonlocal active, max_active
            async with lock:
                active += 1
                max_active = max(max_active, active)
            await asyncio.sleep(sleep_duration)
            async with lock:
                active -= 1
            return STUB_ATTRACTION

        async def stub_debate(self: object, *args: object, **kwargs: object) -> DebateReport:
            nonlocal active, max_active
            async with lock:
                active += 1
                max_active = max(max_active, active)
            await asyncio.sleep(sleep_duration)
            async with lock:
                active -= 1
            return STUB_DEBATE

        async def stub_four_dim(
            self: object, *args: object, **kwargs: object
        ) -> FourDimReviewResult:
            nonlocal active, max_active
            async with lock:
                active += 1
                max_active = max(max_active, active)
            await asyncio.sleep(sleep_duration)
            async with lock:
                active -= 1
            return STUB_FOUR_DIM

        monkeypatch.setattr(f"{engine_mod}.llm_old_reader_analysis", stub_old_reader)
        monkeypatch.setattr(f"{engine_mod}.llm_ai_style_analysis", stub_ai_style)
        monkeypatch.setattr(f"{engine_mod}.attraction_llm_analysis", stub_attraction)
        monkeypatch.setattr(VerdictEngine, "_run_debate", stub_debate)
        monkeypatch.setattr(VerdictEngine, "_run_four_dim_review", stub_four_dim)

        start = time.monotonic()
        verdict = await _run_evaluate()
        elapsed = time.monotonic() - start

        # 并行：5 个维度同时活跃（max_active == 5），总耗时 ≈ 0.3s
        # 串行：每时刻最多 1 个活跃，总耗时 ≈ 1.5s
        assert max_active == 5, f"5 个维度未并行（同时活跃峰值={max_active}）"
        assert elapsed < 1.0, (
            f"evaluate 总耗时 {elapsed:.2f}s，疑似串行执行（并行应≈0.3s，串行≥1.5s）"
        )
        assert verdict is not None

    async def test_evaluate_serial_reference(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """对照实验：桩函数串行 await 时总耗时 ≥ 1.2s（证明耗时上界敏感）。"""
        sleep_duration = 0.3

        async def slow(**kwargs: object) -> LLMOldReaderResult:
            await asyncio.sleep(sleep_duration)
            return STUB_OLD_READER

        start = time.monotonic()
        for _ in range(4):
            await slow()
        elapsed = time.monotonic() - start

        assert elapsed >= 1.0, f"串行基线应≥1.2s，实测 {elapsed:.2f}s（环境时钟异常？）"


# ═══════════════════════════════════════════════════════════════════════════════
#  结果融合验证
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluateResultsMerged:
    """evaluate() 返回值融合验证。"""

    async def test_evaluate_results_merged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """5 个维度固定返回值应被正确融合进 VerdictResult。"""
        _patch_five_dimensions(monkeypatch)

        verdict = await _run_evaluate()

        # final_score 字段存在且为 0-100 浮点分
        assert isinstance(verdict.final_score, float)
        assert 0.0 <= verdict.final_score <= 100.0

        # programmatic_score 来自真实程序化分析（纯计算，0-1 区间）
        assert isinstance(verdict.programmatic_score, float)
        assert 0.0 <= verdict.programmatic_score <= 1.0

        # LLM 老书虫语义分被融合（STUB_OLD_READER.semantic_score=82）
        assert verdict.llm_semantic_score == pytest.approx(82.0)

        # LLM AI味人类相似度被融合（STUB_AI_STYLE.human_like_score=78）
        assert verdict.llm_human_like_score == pytest.approx(78.0)

        # LLM 吸引力专家团队分与建议被融合（STUB_ATTRACTION.attraction_score=74）
        assert verdict.llm_attraction_score == pytest.approx(74.0)
        assert verdict.llm_attraction_fix == "吸引力专家建议：增强开篇钩子"

        # 四维评分与辩论结果进入 VerdictResult
        assert verdict.quality_score == pytest.approx(80.0)
        assert verdict.debate_penalty == pytest.approx(0.0)  # 辩论无问题 → 无惩罚
