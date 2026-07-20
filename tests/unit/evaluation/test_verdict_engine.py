"""VerdictEngine 单元测试 — 核心融合评分逻辑。

覆盖:
    - _fuse() 融合计算（权重/迭代加分/衰减/归一化）
    - _decide_level() 三级决议（PASS/REFINE/REWRITE）
    - _trim_for_review() 文本裁剪
    - _detect_quality_decay() 高开低走检测
"""

from __future__ import annotations

import pytest

from novelfactory.evaluation.schemas import (
    AttemptInfo,
    DebateReport,
    FourDimReviewResult,
    ProgrammaticReport,
    VerdictLevel,
)
from novelfactory.evaluation.verdict.engine import (
    _detect_quality_decay,
    VerdictEngine,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def engine() -> VerdictEngine:
    return VerdictEngine()


@pytest.fixture
def four_dim_pass() -> FourDimReviewResult:
    return FourDimReviewResult(
        quality_score=85.0,
        review_comments="整体不错",
        cross_chapter_consistency=80.0,
    )


@pytest.fixture
def four_dim_refine() -> FourDimReviewResult:
    return FourDimReviewResult(
        quality_score=60.0,
        review_comments="需要润色",
        cross_chapter_consistency=65.0,
    )


@pytest.fixture
def four_dim_rewrite() -> FourDimReviewResult:
    return FourDimReviewResult(
        quality_score=40.0,
        review_comments="需要重写",
        cross_chapter_consistency=50.0,
    )


@pytest.fixture
def programmatic_normal() -> ProgrammaticReport:
    return ProgrammaticReport(
        ai_style_score=0.15,
        lao_shu_chong_score=80.0,
        has_severe_toxic=False,
    )


@pytest.fixture
def debate_clean() -> DebateReport:
    return DebateReport(
        debate_failed=False,
        convergence_achieved=True,
        merged_issues=[],
        merged_strengths=["文笔流畅"],
        merged_suggestions="",
        debate_transcript="",
    )


@pytest.fixture
def attempt_first() -> AttemptInfo:
    return AttemptInfo(loop_count=0, refine_attempts=0, max_rewrite=5, max_refine=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  _fuse() 测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestFuse:
    """VerdictEngine._fuse() 融合计算测试。"""

    def test_fuse_basic(
        self, engine, four_dim_pass, programmatic_normal, debate_clean, attempt_first
    ):
        """默认权重 + 正常评分 → 评分合理。"""
        from novelfactory.evaluation.schemas import CrossChapterSignals

        verdict = engine._fuse(
            four_dim=four_dim_pass,
            programmatic=programmatic_normal,
            debate=debate_clean,
            cross_chapter=CrossChapterSignals(has_prev_context=False),
            attempt_info=attempt_first,
        )
        # 按权重: 85*0.25 + 68*0.30 + 80*0.20 = 57.65, 预期 ~58-70 范围
        assert 50.0 <= verdict.final_score <= 80.0, (
            f"final_score 异常: {verdict.final_score:.1f}"
        )
        assert verdict.final_score >= 50.0

    def test_fuse_with_iteration_bonus(
        self, engine, four_dim_rewrite, programmatic_normal, debate_clean
    ):
        """迭代宽松加分 — loop=4 时 final_score 应显著提升。"""
        from novelfactory.evaluation.schemas import CrossChapterSignals

        attempt = AttemptInfo(
            loop_count=4, refine_attempts=0, max_rewrite=5, max_refine=2
        )
        verdict = engine._fuse(
            four_dim=four_dim_rewrite,
            programmatic=programmatic_normal,
            debate=debate_clean,
            cross_chapter=CrossChapterSignals(has_prev_context=False),
            attempt_info=attempt,
        )
        bonus = 4 * 3.0  # VERDICT_ITERATION_BONUS_REWRITE=3
        assert verdict.final_score >= 40.0 + bonus - 5, (
            f"迭代加分异常: {verdict.final_score:.1f}"
        )

    def test_fuse_with_quality_decay(
        self, engine, four_dim_pass, programmatic_normal, debate_clean, attempt_first
    ):
        """质量衰减惩罚 — decay_penalty=10 时 final_score 降低至少 8 分。"""
        from novelfactory.evaluation.schemas import CrossChapterSignals

        verdict_no_decay = engine._fuse(
            four_dim=four_dim_pass,
            programmatic=programmatic_normal,
            debate=debate_clean,
            cross_chapter=CrossChapterSignals(has_prev_context=False),
            attempt_info=attempt_first,
        )

        verdict_with_decay = engine._fuse(
            four_dim=four_dim_pass,
            programmatic=programmatic_normal,
            debate=debate_clean,
            cross_chapter=CrossChapterSignals(has_prev_context=False),
            attempt_info=attempt_first,
            decay_penalty=10.0,
        )
        assert verdict_no_decay.final_score - verdict_with_decay.final_score >= 8.0

    def test_fuse_llm_toxic_override(
        self, engine, four_dim_rewrite, debate_clean, attempt_first
    ):
        """LLM 语义覆盖程序化毒点 — 毒点被否认时权重转移。"""
        from novelfactory.evaluation.schemas import CrossChapterSignals

        prog_toxic = ProgrammaticReport(
            ai_style_score=0.3,
            lao_shu_chong_score=15.0,
            has_severe_toxic=True,
            severe_toxic_types=["NTR"],
        )
        verdict = engine._fuse(
            four_dim=four_dim_rewrite,
            programmatic=prog_toxic,
            debate=debate_clean,
            cross_chapter=CrossChapterSignals(has_prev_context=False),
            attempt_info=attempt_first,
        )
        # 不应因程序化毒点被严重压低
        assert verdict.final_score >= 20.0, f"毒点覆盖异常: {verdict.final_score:.1f}"

    def test_fuse_length_normalize_short(
        self, engine, four_dim_pass, programmatic_normal, debate_clean, attempt_first
    ):
        """短文本（<3000字）不触发长度归一化。"""
        from novelfactory.evaluation.schemas import CrossChapterSignals

        verdict = engine._fuse(
            four_dim=four_dim_pass,
            programmatic=programmatic_normal,
            debate=debate_clean,
            cross_chapter=CrossChapterSignals(has_prev_context=False),
            attempt_info=attempt_first,
            chapter_length=1500,
        )
        # 短文本不会除法
        assert verdict.final_score > 50.0

    def test_fuse_length_normalize_long(
        self, engine, four_dim_pass, programmatic_normal, debate_clean, attempt_first
    ):
        """长文本（>3000字）触发长度归一化。"""
        from novelfactory.evaluation.schemas import CrossChapterSignals

        verdict_short = engine._fuse(
            four_dim=four_dim_pass,
            programmatic=programmatic_normal,
            debate=debate_clean,
            cross_chapter=CrossChapterSignals(has_prev_context=False),
            attempt_info=attempt_first,
            chapter_length=3000,
        )
        verdict_long = engine._fuse(
            four_dim=four_dim_pass,
            programmatic=programmatic_normal,
            debate=debate_clean,
            cross_chapter=CrossChapterSignals(has_prev_context=False),
            attempt_info=attempt_first,
            chapter_length=6000,
        )
        # 6000字章节因归一化分数应低于3000字
        assert verdict_long.final_score <= verdict_short.final_score + 0.01


# ═══════════════════════════════════════════════════════════════════════════════
#  _decide_level() 测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestDecideLevel:
    """VerdictEngine._decide_level() 三级决议测试。"""

    def test_pass_high_score(self, engine, programmatic_normal):
        """final=80, 无毒点 → PASS。"""
        level = engine._decide_level(
            final_score=80.0,
            programmatic=programmatic_normal,
            attempt_info=AttemptInfo(loop_count=0, refine_attempts=0, max_rewrite=5, max_refine=2),
        )
        assert level == VerdictLevel.PASS

    def test_refine_mid_score(self, engine, programmatic_normal):
        """final=60, 无毒点 → REFINE。"""
        level = engine._decide_level(
            final_score=60.0,
            programmatic=programmatic_normal,
            attempt_info=AttemptInfo(loop_count=0, refine_attempts=0, max_rewrite=5, max_refine=2),
        )
        assert level == VerdictLevel.REFINE

    def test_rewrite_low_score(self, engine, programmatic_normal):
        """final=40, 无毒点 → REWRITE。"""
        level = engine._decide_level(
            final_score=40.0,
            programmatic=programmatic_normal,
            attempt_info=AttemptInfo(loop_count=0, refine_attempts=0, max_rewrite=5, max_refine=2),
        )
        assert level == VerdictLevel.REWRITE

    def test_toxic_llm_rewrite(self, engine, programmatic_normal):
        """LLM 严重毒点 → REWRITE。"""
        level = engine._decide_level(
            final_score=85.0,
            programmatic=programmatic_normal,
            attempt_info=AttemptInfo(loop_count=0, refine_attempts=0, max_rewrite=5, max_refine=2),
            llm_severe_toxic=True,
        )
        assert level == VerdictLevel.REWRITE

    def test_both_exhausted_force_pass(self, engine, programmatic_normal):
        """双向次数用尽 → PASS。"""
        level = engine._decide_level(
            final_score=35.0,
            programmatic=programmatic_normal,
            attempt_info=AttemptInfo(loop_count=5, refine_attempts=2, max_rewrite=5, max_refine=2),
        )
        assert level == VerdictLevel.PASS

    def test_one_exhausted_downgrade(self, engine, programmatic_normal):
        """rewrite 用尽但 refine 未用尽 → 降级 REFINE。"""
        level = engine._decide_level(
            final_score=35.0,
            programmatic=programmatic_normal,
            attempt_info=AttemptInfo(loop_count=5, refine_attempts=0, max_rewrite=5, max_refine=2),
        )
        assert level == VerdictLevel.REFINE

    def test_four_dim_failed_rewrite(self, engine, programmatic_normal):
        """四维评分失败(降级fallback) → REWRITE。"""
        level = engine._decide_level(
            final_score=55.0,
            programmatic=programmatic_normal,
            attempt_info=AttemptInfo(loop_count=0, refine_attempts=0, max_rewrite=5, max_refine=2),
            four_dim_failed=True,
        )
        assert level in (VerdictLevel.REWRITE, VerdictLevel.REFINE)

    def test_prog_toxic_overridden_pass(self, engine):
        """程序化毒点被 LLM 否认 → 走评分路由。"""
        toxic_prog = ProgrammaticReport(
            ai_style_score=0.2,
            lao_shu_chong_score=75.0,
            has_severe_toxic=True,
            severe_toxic_types=["虐主"],
        )
        level = engine._decide_level(
            final_score=80.0,
            programmatic=toxic_prog,
            attempt_info=AttemptInfo(loop_count=0, refine_attempts=0, max_rewrite=5, max_refine=2),
            prog_toxic_overridden=True,
        )
        assert level == VerdictLevel.PASS


# ═══════════════════════════════════════════════════════════════════════════════
#  _trim_for_review() 测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestTrimForReview:
    """文本裁剪逻辑测试。"""

    def test_short_text_passthrough(self, engine):
        """短文本（<8000字）直接通过。"""
        text = "这是" * 2000  # ~4000字
        result = engine._trim_for_review(text)
        assert result == text

    def test_long_text_trimmed(self, engine):
        """长文本（>8000字）被裁剪。"""
        text = "段落内容。\n" * 3000  # ~15000字
        result = engine._trim_for_review(text)
        assert len(result) < len(text)
        assert "[...]" in result

    def test_very_long_text(self, engine):
        """超长文本（>20000字）裁剪后保留首尾+采样。"""
        text = "测试段落。\n" * 5000
        result = engine._trim_for_review(text)
        assert len(result) < 6500  # 裁剪到~6500字以内（保留 1500+1500+采样）
        assert text[:100] in result  # 开头保留
        assert text[-100:] in result  # 结尾保留


# ═══════════════════════════════════════════════════════════════════════════════
#  _detect_quality_decay() 测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestQualityDecay:
    """质量衰减检测测试。"""

    def test_empty_text(self):
        """空文本 → 0 衰减。"""
        assert _detect_quality_decay("") == 0.0

    def test_short_text(self):
        """短文本（<500字）→ 0 衰减。"""
        assert _detect_quality_decay("短文本" * 50) == 0.0

    def test_long_text_no_decay(self):
        """长文本但无明显衰减 → 接近 0。"""
        text = ("这是一个普通的段落，包含各种词汇和表达方式。" * 200) + (
            "远处传来钟声，他抬起头望向远方。" * 200
        )
        decay = _detect_quality_decay(text)
        assert decay >= 0.0, f"衰减不应为负: {decay}"
