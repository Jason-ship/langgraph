"""P1: Circuit breaker — record success/failure, is_open, get_status."""

from __future__ import annotations

from novelfactory.agents.infra.circuit_breaker import (
    circuit_breaker_get_status,
    circuit_breaker_is_open,
    circuit_breaker_record_failure,
    circuit_breaker_record_success,
)


class TestCircuitBreakerSuccess:
    """P1: circuit_breaker_record_success resets failures and opens circuit."""

    def test_success_resets_failures(self):
        circuit_breaker_record_failure("deepseek")
        circuit_breaker_record_failure("deepseek")
        circuit_breaker_record_success("deepseek")
        assert not circuit_breaker_is_open("deepseek")

    def test_success_on_unknown_service_noop(self):
        circuit_breaker_record_success("nonexistent")


class TestCircuitBreakerFailure:
    """P1: circuit_breaker_record_failure increments and opens after threshold."""

    def test_failure_increments(self):
        # reset first
        circuit_breaker_record_success("ark")
        circuit_breaker_record_failure("ark")
        circuit_breaker_record_failure("ark")
        # 2 failures < 20 max → still closed
        assert not circuit_breaker_is_open("ark")

    def test_max_failures_opens_circuit(self):
        # reset
        circuit_breaker_record_success("ark")
        for _ in range(20):
            circuit_breaker_record_failure("ark")
        assert circuit_breaker_is_open("ark")

    def test_unknown_service_noop(self):
        circuit_breaker_record_failure("nonexistent")

    def test_siliconflow_opens_after_10(self):
        circuit_breaker_record_success("siliconflow")
        for _ in range(10):
            circuit_breaker_record_failure("siliconflow")
        assert circuit_breaker_is_open("siliconflow")


class TestCircuitBreakerIsOpen:
    """P1: circuit_breaker_is_open after cooldown closes again."""

    def test_closed_by_default(self):
        assert not circuit_breaker_is_open("matrix")

    def test_unknown_service_false(self):
        assert not circuit_breaker_is_open("nonexistent")


class TestCircuitBreakerGetStatus:
    """P1: circuit_breaker_get_status returns structured dict."""

    def test_status_has_all_services(self):
        status = circuit_breaker_get_status()
        assert "deepseek" in status
        assert "ark" in status
        assert "siliconflow" in status

    def test_status_keys(self):
        status = circuit_breaker_get_status()
        for svc in ("deepseek",):
            svc_status = status[svc]
            assert "open" in svc_status
            assert "failures" in svc_status
            assert "max_failures" in svc_status
            assert "cooldown_remaining" in svc_status
