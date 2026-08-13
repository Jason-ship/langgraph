"""统一评审系统单元测试（schema / 解析器 / 自洽校验 / 仲裁 / 引擎）。"""

from __future__ import annotations

import asyncio

from novelfactory.evaluation.unified.parser import (
    apply_consistency_check,
    parse_review_output,
)
from novelfactory.evaluation.unified.schemas import UnifiedFourDim, UnifiedReviewResult

SAMPLE = """<review_analysis>毒点核查：P8 战力突兀。</review_analysis>
[评分] final=78.5
[四维-剧情逻辑] 26/30
[四维-文笔] 20/25
[四维-人物] 21/25
[四维-世界观] 17/20
[毒点] POWER_BREAK|P8|severe|战力突兀
[爽点] 打脸|P5; 升级|P14
[AI味] 0.72
[吸引力] 85  [沉浸] 88
[跨章] 82
[衰减] none
[废话段] P3; P12
[跳跃] P8->P9|场景无过渡

评审意见：
- 老书虫视角：……
- 番茄编辑视角：……
- 读者视角：……
- 评论员视角：……
- 分歧点：老书虫认为 P8 战力突兀，番茄编辑认为可接受
- 修复指令：P3 删冗余环境描写
"""


def test_four_dim_total():
    fd = UnifiedFourDim(logic=26.0, writing=20.0, character=21.0, world=17.0)
    assert fd.total() == 84.0


def test_review_result_defaults():
    r = UnifiedReviewResult()
    assert r.failed is False
    assert r.severe_toxic is False
    assert r.final_score == 60.0


def test_parse_review_output():
    r = parse_review_output(SAMPLE)
    assert r is not None
    assert r.final_score == 78.5
    assert r.four_dim.logic == 26.0
    assert r.severe_toxic is True
    assert r.water_paragraphs == [3, 12]
    assert r.shuangdian_count == 2
    assert len(r.perspective_disagreements) == 1
    assert r.attraction_score == 85.0
    assert r.immersion_score == 88.0
    assert r.cross_chapter_score == 82.0


def test_parse_no_toxic():
    raw = SAMPLE.replace("[毒点] POWER_BREAK|P8|severe|战力突兀", "[毒点] 无")
    r = parse_review_output(raw)
    assert r is not None
    assert r.toxic_points == []
    assert r.severe_toxic is False


def test_consistency_caps_severe_toxic():
    r = parse_review_output(SAMPLE)
    changed = apply_consistency_check(r)
    assert changed is True
    assert r.final_score <= 70.0


def test_consistency_caps_no_shuangdian():
    raw = SAMPLE.replace("[爽点] 打脸|P5; 升级|P14", "[爽点] 无").replace(
        "[评分] final=78.5", "[评分] final=80.0"
    )
    r = parse_review_output(raw)
    changed = apply_consistency_check(r)
    assert changed is True
    assert r.final_score <= 65.0


def test_consistency_rebases_to_four_dim():
    raw = SAMPLE.replace("[评分] final=78.5", "[评分] final=95.0")
    r = parse_review_output(raw)
    changed = apply_consistency_check(r)
    assert changed is True
    assert abs(r.final_score - r.four_dim.total()) <= 15.0


def test_parse_garbage_returns_none():
    assert parse_review_output("完全不是评审输出") is None


def test_parse_empty_returns_none():
    assert parse_review_output("") is None


# ── Task 3: prompts ────────────────────────────────────────────────────────

from novelfactory.evaluation.unified.prompts import (  # noqa: E402
    build_arbitration_prompt,
    build_quick_recheck_prompt,
    build_unified_review_prompt,
)


def test_build_unified_prompt_contains_core_directives():
    p = build_unified_review_prompt(
        chapter_text="测试正文" * 50, genre="都市", prev_summary="前情", guide="题材指南"
    )
    for keyword in (
        "[评分] final=", "老书虫视角", "番茄编辑视角", "读者视角", "评论员视角",
        "[废话段]", "[爽点]", "网文爽感铁律", "题材评分指引", "前章摘要",
    ):
        assert keyword in p


def test_build_recheck_prompt_injects_old_issues():
    p = build_quick_recheck_prompt(
        chapter_text="正文", old_issues=["P8 战力突兀", "P3 水文"], old_score=68.0
    )
    assert "P8 战力突兀" in p
    assert "P3 水文" in p
    assert "68" in p


def test_build_arbitration_prompt_injects_disagreements():
    p = build_arbitration_prompt(
        disagreements=["severe: P8 战力突兀（老书虫 vs 番茄编辑）"], old_score=78.5
    )
    assert "P8 战力突兀" in p
    assert "78.5" in p


# ── Task 4A: 仲裁模块 ──────────────────────────────────────────────────────

from novelfactory.evaluation.unified.arbitration import (  # noqa: E402
    arbitrate,
    parse_arbitration,
)

ARB = """<review_analysis>权衡：老书虫看重战力一致，番茄编辑看重节奏。</review_analysis>
[裁决] 采纳老书虫：P8 需补战力铺垫，但不必重写整章
[分数修正] final: 78.5 -> 74.0
[备注] REFINE"""


def test_parse_arbitration():
    r = parse_arbitration(ARB)
    assert r is not None
    assert r["new_score"] == 74.0
    assert r["action"] == "REFINE"


def test_parse_arbitration_garbage():
    assert parse_arbitration("乱码") is None


class _FakeArbLlm:
    async def ainvoke(self, *a, **kw):
        return type("R", (), {"content": ARB})()


def test_arbitrate_with_llm():
    score = asyncio.run(
        arbitrate(_FakeArbLlm(), disagreements=["severe: P8 战力突兀"], old_score=78.5)
    )
    assert score == 74.0


def test_arbitrate_empty_returns_old():
    score = asyncio.run(arbitrate(_FakeArbLlm(), disagreements=[], old_score=78.5))
    assert score == 78.5


# ── Task 4: 统一评审引擎 ────────────────────────────────────────────────────

from novelfactory.evaluation.unified.engine import UnifiedReviewEngine  # noqa: E402

OK_REVIEW = """<review_analysis>ok</review_analysis>
[评分] final=82.0
[四维-剧情逻辑] 27/30
[四维-文笔] 21/25
[四维-人物] 22/25
[四维-世界观] 18/20
[毒点] 无
[爽点] 打脸|P5; 升级|P14; 奖励|P20
[AI味] 0.75
[吸引力] 88  [沉浸] 90
[跨章] 84
[衰减] none
[废话段] 无
[跳跃] 无
评审意见：
- 老书虫视角：爽点密
- 番茄编辑视角：钩子强
- 读者视角：沉浸
- 评论员视角：逻辑通
- 分歧点：无
- 修复指令：P5 补一句反派反应"""


class _FakeLlm:
    def __init__(self, payload):
        self._payload = payload

    async def ainvoke(self, *a, **kw):
        return type("R", (), {"content": self._payload})()


def test_engine_evaluate_success():
    engine = UnifiedReviewEngine(_FakeLlm(OK_REVIEW))
    r = asyncio.run(engine.evaluate(chapter_text="正文", genre="都市", prev_summary="", guide=""))
    assert r.failed is False
    assert r.final_score == 82.0
    assert r.shuangdian_count == 3


def test_engine_evaluate_retry_then_fallback():
    engine = UnifiedReviewEngine(_FakeLlm("不可解析"))
    r = asyncio.run(
        engine.evaluate(chapter_text="正文", genre="都市", prev_summary="", guide="", retries=1)
    )
    assert r.failed is True
    assert r.final_score == 60.0  # fallback
    assert r.retried is True


def test_engine_evaluate_arbitrates_severe_disagreement():
    payload = OK_REVIEW.replace(
        "分歧点：无", "分歧点：severe 老书虫认为 P8 战力突兀，番茄编辑认为可接受"
    )
    engine = UnifiedReviewEngine(_FakeLlm(payload))
    # _FakeLlm 对仲裁调用也返回同一 payload；parse_arbitration 提取不到 [分数修正] → 返回原分
    r = asyncio.run(engine.evaluate(chapter_text="正文", genre="都市", prev_summary="", guide=""))
    assert r.failed is False
    assert r.perspective_disagreements  # 仲裁被触发但原分保留（mock 无 [分数修正]）


def test_engine_quick_recheck():
    engine = UnifiedReviewEngine(_FakeLlm(OK_REVIEW))
    r = asyncio.run(
        engine.quick_recheck(chapter_text="正文", old_issues=["P8 战力突兀"], old_score=68.0)
    )
    assert r.failed is False
    assert r.final_score == 82.0


# 复查 prompt 输出格式（含 [评分] final=，与 parse_review_output 对齐）——
# 防止 quick_recheck 解析器与复查输出格式再次错位（回归防护）
RECHECK_PAYLOAD = """<review_analysis>P8 战力铺垫已补，未引入新问题。</review_analysis>
[复查] passed=true
[评分] final=80.0
[毒点] 无
[新增问题] 无
[分数变化] final: 68.0 -> 80.0"""


def test_parse_recheck_payload():
    """复查输出应能被统一解析器正常解析（评分标签对齐）。"""
    r = parse_review_output(RECHECK_PAYLOAD)
    assert r is not None
    assert r.final_score == 80.0
    assert r.toxic_points == []


def test_engine_quick_recheck_with_recheck_payload():
    """quick_recheck 对复查格式输出应解析成功而非 fallback。"""
    engine = UnifiedReviewEngine(_FakeLlm(RECHECK_PAYLOAD))
    r = asyncio.run(
        engine.quick_recheck(chapter_text="正文", old_issues=["P8 战力突兀"], old_score=68.0)
    )
    assert r.failed is False
    assert r.final_score == 80.0


# ── v8.2 迁移自 debate 的 Markdown 分段解析（critic_pre 前置评估依赖） ──

from novelfactory.evaluation.utils import parse_markdown_sections  # noqa: E402


def test_parse_markdown_sections_lists():
    """issues/strengths 列表分支应正确提取（回归防护：_extract_list_items 迁移完整性）。"""
    raw = """## 评审意见
整体可以。
## 问题列表
- P8 战力突兀
- P12 水文
## 亮点
1. 钩子设置好
2. 爽点密度高
## 改进建议
P8 补铺垫"""
    r = parse_markdown_sections(raw)
    assert "整体可以" in r["review_comments"]
    assert r["issues"] == ["P8 战力突兀", "P12 水文"]
    assert r["strengths"] == ["钩子设置好", "爽点密度高"]
    assert "P8 补铺垫" in r["suggestions"]


def test_parse_markdown_sections_json_fallback():
    """JSON 输入兜底解析。"""
    r = parse_markdown_sections('{"review_comments": "ok", "issues": ["a"]}')
    assert r["review_comments"] == "ok"
    assert r["issues"] == ["a"]
