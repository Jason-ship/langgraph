"""LLM 语义分析模块测试 — 老书虫 + AI味。

覆盖:
    - LLM 输出 JSON 解析
    - 降级路径
"""

from __future__ import annotations

import pytest

from novelfactory.agents.infra.serialization import validate_json_output


class TestLLMOldReaderParsing:
    """LLM 老书虫 JSON 解析测试。"""

    def test_valid_json(self):
        """完整字段的 JSON → 正确解析。"""
        raw = """
        {
            "semantic_score": 85.0,
            "review_comments": "整体不错",
            "has_severe_toxic": false,
            "toxic_details": ["行文略平"],
            "implicit_toxic_found": false
        }
        """
        parsed, err = validate_json_output(
            raw,
            required_keys=["semantic_score", "review_comments"],
            fail_closed=False,
        )
        assert parsed is not None, f"解析失败: {err}"
        assert float(parsed["semantic_score"]) == 85.0
        assert parsed["has_severe_toxic"] is False

    def test_partial_json_with_code_block(self):
        """代码块包裹的 JSON → 正确提取。"""
        raw = '```json\n{"semantic_score": 70.0, "review_comments": "还可以"}\n```'
        parsed, err = validate_json_output(
            raw,
            required_keys=["semantic_score", "review_comments"],
            fail_closed=False,
        )
        assert parsed is not None, f"解析失败: {err}"
        assert float(parsed["semantic_score"]) == 70.0

    def test_missing_required_field(self):
        """缺少必需字段 → err 不为空。"""
        raw = '{"semantic_score": 80.0}'
        parsed, err = validate_json_output(
            raw,
            required_keys=["semantic_score", "review_comments"],
            fail_closed=True,
        )
        assert err is not None, "应有解析错误"

    def test_invalid_json(self):
        """非 JSON 响应 → 降级路径。"""
        raw = "章节质量不错，评分85分。没有其他问题。"
        parsed, err = validate_json_output(
            raw,
            required_keys=["semantic_score"],
            fail_closed=False,
        )
        assert err is not None or parsed is not None


class TestLLMAIStyleParsing:
    """LLM AI味语义分析 JSON 解析测试。"""

    def test_valid_json(self):
        """完整字段 JSON → 正确解析。"""
        raw = """
        {
            "human_like_score": 75.0,
            "ai_tells": ["描写模板化"],
            "natural_expressions": ["对话自然"],
            "overall_assessment": "有一定AI痕迹",
            "has_ai_style_issues": true
        }
        """
        parsed, err = validate_json_output(
            raw,
            required_keys=["human_like_score", "overall_assessment"],
            fail_closed=False,
        )
        assert parsed is not None, f"解析失败: {err}"
        assert float(parsed["human_like_score"]) == 75.0

    def test_ai_style_score_range(self):
        """human_like_score 解析后应按原始值返回（范围验证在应用层）。"""
        raw = '{"human_like_score": 120.0, "overall_assessment": "test"}'
        parsed, err = validate_json_output(
            raw,
            required_keys=["human_like_score", "overall_assessment"],
            fail_closed=False,
        )
        assert parsed is not None
        score = float(parsed["human_like_score"])
        # validate_json_output 不做范围裁剪，120.0 按原值返回
        assert score == 120.0

    def test_missing_optional_fields(self):
        """缺少可选字段不应导致解析失败。"""
        raw = '{"human_like_score": 65.0, "overall_assessment": "ok"}'
        parsed, err = validate_json_output(
            raw,
            required_keys=["human_like_score", "overall_assessment"],
            fail_closed=False,
        )
        assert parsed is not None
