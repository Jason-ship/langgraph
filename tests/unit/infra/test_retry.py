"""llm_call_with_retry retry/timeout + error classification tests."""

from __future__ import annotations

from novelfactory.agents.infra._retry_common import (
    _categorize_status,
    _classify_error,
    _extract_http_status,
    _extract_model_name,
    _resolve_retry_policy,
)


class TestRetryPolicyResolution:
    """Test the new RetryPolicy mapping from checkpointer.py."""

    def test_resolve_default_policy(self):
        policy = _resolve_retry_policy("default")
        assert policy.max_attempts == 3

    def test_resolve_writer_policy(self):
        policy = _resolve_retry_policy("writer")
        assert policy.max_attempts == 5

    def test_resolve_reviewer_policy(self):
        policy = _resolve_retry_policy("reviewer")
        assert policy.max_attempts == 3

    def test_resolve_unknown_fallback_to_default(self):
        policy = _resolve_retry_policy("unknown")
        assert policy is not None
        assert hasattr(policy, "max_attempts")


class TestCategorizeStatus:
    """P1: _categorize_status — HTTP status → retry strategy."""

    def test_400_no_retry(self):
        assert _categorize_status(400) == "no_retry"

    def test_401_no_retry(self):
        assert _categorize_status(401) == "no_retry"

    def test_403_no_retry(self):
        assert _categorize_status(403) == "no_retry"

    def test_429_immediate(self):
        assert _categorize_status(429) == "immediate"

    def test_500_backoff(self):
        assert _categorize_status(500) == "backoff"

    def test_502_backoff(self):
        assert _categorize_status(502) == "backoff"

    def test_503_backoff(self):
        assert _categorize_status(503) == "backoff"

    def test_504_backoff(self):
        assert _categorize_status(504) == "backoff"

    def test_unknown_status_backoff(self):
        assert _categorize_status(418) == "backoff"


class TestExtractHttpStatus:
    """P1: _extract_http_status — extract status from exception."""

    class FakeHttpError(Exception):
        def __init__(self, status_code):
            self.status_code = status_code

    class FakeNestedError(Exception):
        def __init__(self, status_code):
            self.response = type("obj", (), {"status_code": status_code})()

    def test_extracts_status_from_attribute(self):
        exc = self.FakeHttpError(503)
        assert _extract_http_status(exc) == 503

    def test_extracts_from_response_attribute(self):
        exc = self.FakeNestedError(429)
        assert _extract_http_status(exc) == 429

    def test_no_status_returns_none(self):
        exc = Exception("generic")
        assert _extract_http_status(exc) is None


class TestClassifyError:
    """P1: _classify_error — exception → (category, http_status)."""

    def test_timeout_classified_as_backoff(self):
        from novelfactory.agents.infra.timeout import LLMTimeoutError
        category, status = _classify_error(LLMTimeoutError("timeout"))
        assert category == "backoff"
        assert status is None

    def test_oserror_classified_as_backoff(self):
        category, status = _classify_error(OSError("connection"))
        assert category == "backoff"
        assert status is None

    def test_generic_exception_classified_as_backoff(self):
        category, status = _classify_error(Exception("unknown"))
        assert category == "backoff"
        assert status is None

    def test_classify_with_http_status(self):
        class FakeStatusError(Exception):
            status_code = 429

        category, status = _classify_error(FakeStatusError())
        assert category == "immediate"
        assert status == 429


class TestExtractModelName:
    """P1: _extract_model_name — extract model from callable."""

    class FakeFunc:
        model = "deepseek-chat"

    def test_extracts_from_func_attribute(self):
        func = self.FakeFunc()
        assert _extract_model_name(func) == "deepseek-chat"

    def test_extracts_from_kwargs(self):
        def noop():
            pass
        assert _extract_model_name(noop, model="m2.7") == "m2.7"

    def test_fallback_for_plain_func(self):
        def noop():
            pass
        result = _extract_model_name(noop)
        assert result == "deepseek-v4-flash"
