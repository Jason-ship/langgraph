"""P1: JSON extraction and validation from LLM output text."""

from __future__ import annotations

from novelfactory.agents.infra.serialization import (
    _extract_json_from_text,
    _sanitize_control_chars,
    validate_json_output,
)


class TestExtractJsonFromText:
    """P1: _extract_json_from_text — JSON extraction from LLM output."""

    def test_extracts_json_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = _extract_json_from_text(text)
        assert result == {"key": "value"}

    def test_extracts_plain_code_block(self):
        text = '```\n{"key": "value"}\n```'
        result = _extract_json_from_text(text)
        assert result == {"key": "value"}

    def test_extracts_raw_json_object(self):
        text = '{"key": "value", "num": 42}'
        result = _extract_json_from_text(text)
        assert result == {"key": "value", "num": 42}

    def test_extracts_nested_json(self):
        text = '{"outer": {"inner": [1, 2, 3]}}'
        result = _extract_json_from_text(text)
        assert result == {"outer": {"inner": [1, 2, 3]}}

    def test_returns_none_for_empty_text(self):
        assert _extract_json_from_text("") is None
        assert _extract_json_from_text(None) is None

    def test_returns_none_for_invalid_json(self):
        text = '{"key": "value", invalid}'
        assert _extract_json_from_text(text) is None

    def test_returns_none_for_plain_text(self):
        text = "This is just some plain text without JSON."
        assert _extract_json_from_text(text) is None

    def test_extracts_json_amidst_text(self):
        text = 'Some commentary: {"status": "ok"} and more text'
        result = _extract_json_from_text(text)
        assert result == {"status": "ok"}

    def test_sanitizes_control_characters(self):
        text = '{"k": "v\u0001alue"}'
        result = _extract_json_from_text(text)
        assert result == {"k": "v alue"}


class TestValidateJsonOutput:
    """P1: validate_json_output — fail-closed validation."""

    def test_validates_with_required_keys(self):
        raw = '{"quality_score": 85.0, "review_comments": "Good"}'
        parsed, err = validate_json_output(
            raw, required_keys=["quality_score", "review_comments"]
        )
        assert parsed is not None
        assert parsed["quality_score"] == 85.0
        assert err == ""

    def test_fail_closed_missing_keys(self):
        raw = '{"quality_score": 85.0}'
        parsed, err = validate_json_output(
            raw, required_keys=["quality_score", "review_comments"], fail_closed=True
        )
        assert parsed is None
        assert "Missing required keys" in err

    def test_fail_open_accepts_missing_keys(self):
        raw = '{"quality_score": 85.0}'
        parsed, err = validate_json_output(
            raw, required_keys=["quality_score", "review_comments"], fail_closed=False
        )
        assert parsed is not None
        assert parsed["quality_score"] == 85.0

    def test_parse_failure_returns_none(self):
        raw = "not json at all"
        parsed, err = validate_json_output(raw, required_keys=["x"])
        assert parsed is None
        assert "JSON parse failed" in err


class TestSanitizeControlChars:
    """Utility: _sanitize_control_chars."""

    def test_passthrough_normal_text(self):
        assert _sanitize_control_chars("hello world") == "hello world"

    def test_replaces_null_byte(self):
        result = _sanitize_control_chars("a\x00b")
        assert result == "a b"

    def test_replaces_tab(self):
        result = _sanitize_control_chars("a\tb")
        assert "\t" not in result

    def test_preserves_unicode_chinese(self):
        assert _sanitize_control_chars("你好世界") == "你好世界"
