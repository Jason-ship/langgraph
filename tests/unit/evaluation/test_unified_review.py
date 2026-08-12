"""统一评审系统单元测试（schema / 解析器 / 自洽校验 / 仲裁 / 引擎）。"""

from __future__ import annotations

import asyncio

from novelfactory.evaluation.unified.schemas import UnifiedFourDim, UnifiedReviewResult
from novelfactory.evaluation.unified.parser import parse_review_output, apply_consistency_check

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
