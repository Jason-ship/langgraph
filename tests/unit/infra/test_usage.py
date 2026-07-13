"""P1: Usage tracking — token extraction, reset/read accumulator, count_tokens."""

from __future__ import annotations

from novelfactory.agents.infra.usage import (
    _extract_tokens_from_dict,
    _extract_usage_from_result,
    _try_extract_usage,
    count_tokens,
    count_tokens_dict,
    read_usage_tracking,
    reset_usage_tracking,
)


class TestExtractTokensFromDict:
    """P1: _extract_tokens_from_dict — both naming conventions."""

    def test_openai_style(self):
        p, c = _extract_tokens_from_dict({"prompt_tokens": 100, "completion_tokens": 50})
        assert p == 100
        assert c == 50

    def test_anthropic_style(self):
        p, c = _extract_tokens_from_dict({"input_tokens": 200, "output_tokens": 100})
        assert p == 200
        assert c == 100

    def test_empty_dict_returns_zeros(self):
        p, c = _extract_tokens_from_dict({})
        assert p == 0
        assert c == 0


class TestTryExtractUsage:
    """P1: _try_extract_usage — LangChain result object patterns."""

    def test_none_returns_none(self):
        assert _try_extract_usage(None) is None

    def test_response_metadata_usage(self):
        obj = type("obj", (), {"response_metadata": {"usage": {"prompt_tokens": 10}}})()
        result = _try_extract_usage(obj)
        assert result == {"prompt_tokens": 10}

    def test_usage_metadata_direct(self):
        obj = type("obj", (), {"usage_metadata": {"prompt_tokens": 20}})()
        result = _try_extract_usage(obj)
        assert result == {"prompt_tokens": 20}

    def test_dict_with_usage_key(self):
        result = _try_extract_usage({"usage": {"prompt_tokens": 30}})
        assert result == {"prompt_tokens": 30}


class TestExtractUsageFromResult:
    """P1: _extract_usage_from_result — pull tokens from result."""

    def test_extracts_from_valid_result(self):
        obj = type("obj", (), {
            "usage_metadata": {"prompt_tokens": 100, "completion_tokens": 50},
        })()
        p, c = _extract_usage_from_result(obj)
        assert p == 100
        assert c == 50

    def test_empty_result_returns_zeros(self):
        p, c = _extract_usage_from_result({})
        assert p == 0
        assert c == 0


class TestCountTokens:
    """P1: count_tokens — regex heuristic fallback."""

    def test_empty_string_zero(self):
        assert count_tokens("") == 0

    def test_english_text(self):
        tokens = count_tokens("hello world")
        assert tokens > 0

    def test_chinese_text(self):
        tokens = count_tokens("你好世界")
        assert tokens > 0

    def test_mixed_text(self):
        tokens = count_tokens("hello 你好 world 世界")
        assert tokens > 0


class TestCountTokensDict:
    """P1: count_tokens_dict — sum across dict values."""

    def test_simple_dict(self):
        data = {"a": "hello", "b": "world"}
        tokens = count_tokens_dict(data)
        assert tokens > 0

    def test_nested_dict(self):
        data = {"outer": {"inner": "hello world"}}
        tokens = count_tokens_dict(data)
        assert tokens > 0

    def test_list_value(self):
        data = {"items": ["hello", "world", "test"]}
        tokens = count_tokens_dict(data)
        assert tokens > 0


class TestUsageTracker:
    """P1: reset_usage_tracking + read_usage_tracking cycle."""

    def test_reset_then_read_empty(self):
        reset_usage_tracking()
        snapshot = read_usage_tracking()
        assert snapshot["prompt_tokens"] == 0
        assert snapshot["completion_tokens"] == 0
        assert snapshot["total_tokens"] == 0
        assert snapshot["estimated_cost_cny"] == 0.0
        assert snapshot["calls"] == []

    def test_read_after_reset_returns_empty(self):
        reset_usage_tracking()
        result = read_usage_tracking()
        assert result["total_tokens"] == 0
        assert result["estimated_cost_cny"] == 0.0
