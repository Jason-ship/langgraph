"""P0: _last_value reducer + _add_usage dedup reducer + _chapter_key tests.

These reducers are critical for production correctness — the _last_value
reducer fixed v3/v7 InvalidUpdateError bugs, and _add_usage dedup prevents
double-counting token usage on chapter rewrites.
"""

from __future__ import annotations

from novelfactory.state.novel_state import (
    _add_usage,
    _chapter_key,
    _last_value,
)


class TestLastValueReducer:
    """P0: _last_value — last-write-wins reducer for multi-source scalar fields."""

    def test_update_wins_when_not_none(self):
        """New value should replace existing when not None."""
        assert _last_value("old", "new") == "new"
        assert _last_value(0, 42) == 42
        assert _last_value(False, True) is True

    def test_preserves_existing_when_update_is_none(self):
        """None update should preserve the existing value."""
        assert _last_value("existing", None) == "existing"
        assert _last_value(42, None) == 42

    def test_both_none_returns_none(self):
        """Both None should return None."""
        assert _last_value(None, None) is None

    def test_empty_string_is_not_none(self):
        """Empty string is not None, so it should win."""
        assert _last_value("existing", "") == ""

    def test_zero_is_not_none(self):
        """0 is not None, so it should win."""
        assert _last_value(42, 0) == 0


class TestChapterKey:
    """_chapter_key builds dedup keys for usage records."""

    def test_normal_chapter_key(self):
        assert _chapter_key({"chapter_number": 3, "phase": "writing"}) == "ch3_writing"

    def test_setup_phase_key(self):
        assert _chapter_key({"chapter_number": 0, "phase": "setup"}) == "ch0_setup"

    def test_missing_phase_defaults_to_unknown(self):
        assert _chapter_key({"chapter_number": 5}) == "ch5_unknown"

    def test_missing_chapter_defaults_to_zero(self):
        assert _chapter_key({"phase": "writing"}) == "ch0_writing"


class TestAddUsageReducer:
    """P0: _add_usage — merge total_usage dicts with dedup on (ch, phase)."""

    def test_empty_incoming_returns_existing(self):
        existing = {"chapter_usages": [{"chapter_number": 1, "phase": "writing",
                                         "prompt_tokens": 100, "completion_tokens": 50}]}
        assert _add_usage(existing, {}) == existing

    def test_empty_existing_returns_incoming(self):
        """Empty existing: incoming data passed through with totals computed."""
        incoming = {"chapter_usages": [{"chapter_number": 1, "phase": "writing",
                                        "prompt_tokens": 100, "completion_tokens": 50}]}
        result = _add_usage({}, incoming)
        # incoming data preserved with dedup + sort + computed totals
        assert len(result["chapter_usages"]) == 1
        assert result["chapter_usages"][0]["prompt_tokens"] == 100
        assert result["total_tokens"] == 150  # computed from chapter_usages

    def test_dedup_replaces_same_chapter_phase(self):
        """Chapter rewrite (same chapter_number + phase) replaces old record."""
        existing = {
            "chapter_usages": [
                {"chapter_number": 3, "phase": "writing",
                 "prompt_tokens": 500, "completion_tokens": 300},
            ],
        }
        incoming = {
            "chapter_usages": [
                {"chapter_number": 3, "phase": "writing",
                 "prompt_tokens": 600, "completion_tokens": 400},
            ],
        }
        result = _add_usage(existing, incoming)
        usages = result["chapter_usages"]
        assert len(usages) == 1
        assert usages[0]["prompt_tokens"] == 600  # newer wins
        assert usages[0]["completion_tokens"] == 400
        # Totals recomputed from chapter_usages
        assert result["total_tokens"] == 1000

    def test_appends_different_phase(self):
        """Same chapter, different phase → appended (not replaced)."""
        existing = {
            "chapter_usages": [
                {"chapter_number": 3, "phase": "writing",
                 "prompt_tokens": 500, "completion_tokens": 300},
            ],
        }
        incoming = {
            "chapter_usages": [
                {"chapter_number": 3, "phase": "refine",
                 "prompt_tokens": 200, "completion_tokens": 100},
            ],
        }
        result = _add_usage(existing, incoming)
        usages = result["chapter_usages"]
        assert len(usages) == 2
        assert result["total_tokens"] == 500 + 300 + 200 + 100

    def test_appends_new_chapter(self):
        existing = {
            "chapter_usages": [
                {"chapter_number": 1, "phase": "writing",
                 "prompt_tokens": 100, "completion_tokens": 50},
            ],
        }
        incoming = {
            "chapter_usages": [
                {"chapter_number": 2, "phase": "writing",
                 "prompt_tokens": 200, "completion_tokens": 100},
            ],
        }
        result = _add_usage(existing, incoming)
        assert len(result["chapter_usages"]) == 2
        assert result["total_tokens"] == 150 + 300

    def test_totals_recomputed_from_merged_usages(self):
        """Totals are recomputed from dedup'd chapter_usages, not summed."""
        existing = {
            "chapter_usages": [
                {"chapter_number": 1, "phase": "writing",
                 "prompt_tokens": 1000, "completion_tokens": 500},
            ],
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
        }
        incoming = {
            "chapter_usages": [
                {"chapter_number": 1, "phase": "writing",
                 "prompt_tokens": 200, "completion_tokens": 100},
            ],
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "total_tokens": 300,
        }
        result = _add_usage(existing, incoming)
        # The rewrite replaces ch1_writing: 1000+500 → 200+100
        # Totals should be 300, NOT 1500+300=1800
        assert result["total_tokens"] == 300
        assert result["prompt_tokens"] == 200
        assert result["completion_tokens"] == 100

    def test_merges_model_breakdown(self):
        existing = {
            "chapter_usages": [],
            "model_breakdown": {
                "deepseek-chat": {"prompt_tokens": 100, "completion_tokens": 50,
                                  "total_tokens": 150, "estimated_cost_cny": 0.0002},
            },
        }
        incoming = {
            "chapter_usages": [],
            "model_breakdown": {
                "deepseek-chat": {"prompt_tokens": 200, "completion_tokens": 100,
                                  "total_tokens": 300, "estimated_cost_cny": 0.0004},
            },
        }
        result = _add_usage(existing, incoming)
        # Latest per model wins
        assert result["model_breakdown"]["deepseek-chat"]["prompt_tokens"] == 200

    def test_sorts_usages_by_chapter_then_phase(self):
        incoming = {
            "chapter_usages": [
                {"chapter_number": 3, "phase": "writing",
                 "prompt_tokens": 300, "completion_tokens": 150},
                {"chapter_number": 1, "phase": "refine",
                 "prompt_tokens": 200, "completion_tokens": 100},
                {"chapter_number": 1, "phase": "writing",
                 "prompt_tokens": 100, "completion_tokens": 50},
            ],
        }
        result = _add_usage({}, incoming)
        usages = result["chapter_usages"]
        assert usages[0]["chapter_number"] == 1
        assert usages[0]["phase"] == "refine"
        assert usages[1]["chapter_number"] == 1
        assert usages[1]["phase"] == "writing"
        assert usages[2]["chapter_number"] == 3
