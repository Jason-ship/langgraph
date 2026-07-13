"""P1: Quota check logic — threshold functions and snapshot evaluation."""

from __future__ import annotations

from novelfactory.agents.infra.quota import (
    _format_reset_time,
    _quota_block_threshold,
    _quota_check_before_call,
    _quota_check_interval,
    _quota_warn_threshold,
)


class TestQuotaThresholdDefaults:
    """P1: Default quota threshold values (when config unavailable)."""

    def test_default_check_interval(self):
        assert _quota_check_interval() == 60.0

    def test_default_warn_threshold(self):
        assert _quota_warn_threshold() == 20.0

    def test_default_block_threshold(self):
        assert _quota_block_threshold() == 5.0

    def test_default_check_before_call(self):
        assert _quota_check_before_call() is False


class TestFormatResetTime:
    """P1: _format_reset_time — human-readable duration."""

    def test_zero_seconds(self):
        assert _format_reset_time(0) == "unknown"

    def test_negative_seconds(self):
        assert _format_reset_time(-10) == "unknown"

    def test_minutes_only(self):
        assert _format_reset_time(300) == "5m"

    def test_hours_and_minutes(self):
        assert _format_reset_time(5400) == "1h 30m"

    def test_large_duration(self):
        result = _format_reset_time(86400)
        assert result == "24h 0m"
