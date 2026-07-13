"""Per-service circuit breaker for LLM providers (ARK/DeepSeek/硅基流动)."""

from __future__ import annotations

import time as _time_module

from novelfactory.agents.infra.logger import get_logger
from novelfactory.config.constants import (
    CIRCUIT_BREAKER_CONFIG as _CIRCUIT_BREAKER_CONFIG,
)

# v5.1.1: 完善熔断器，覆盖实际使用的 LLM provider (ARK/DeepSeek/硅基流动)。
# "matrix" 保留为兼容历史配置。
# v6.1: 配置统一从 config.constants 导入（唯一真实来源）。

_circuit_state: dict = {
    name: {"failures": 0, "last_failure_ts": 0.0, "open": False}
    for name in _CIRCUIT_BREAKER_CONFIG
}


def circuit_breaker_record_success(service: str) -> None:
    if service in _circuit_state:
        _circuit_state[service]["failures"] = 0
        _circuit_state[service]["open"] = False


def circuit_breaker_record_failure(service: str) -> None:
    if service not in _circuit_state or service not in _CIRCUIT_BREAKER_CONFIG:
        return
    state = _circuit_state[service]
    cfg = _CIRCUIT_BREAKER_CONFIG[service]
    state["failures"] += 1
    state["last_failure_ts"] = _time_module.time()
    if state["failures"] >= cfg["max_failures"]:
        state["open"] = True
        get_logger("novelfactory.circuit_breaker").warning(
            "[circuit_breaker] service='%s' OPEN after %d failures — fast-failing for %ds",
            service,
            state["failures"],
            cfg["cooldown_seconds"],
        )


def circuit_breaker_is_open(service: str) -> bool:
    if service not in _circuit_state or service not in _CIRCUIT_BREAKER_CONFIG:
        return False
    state = _circuit_state[service]
    cfg = _CIRCUIT_BREAKER_CONFIG[service]
    if not state["open"]:
        return False
    elapsed = _time_module.time() - state["last_failure_ts"]
    if elapsed >= cfg["cooldown_seconds"]:
        state["open"] = False
        state["failures"] = 0
        get_logger("novelfactory.circuit_breaker").info(
            "[circuit_breaker] service='%s' cooldown expired — probing",
            service,
        )
        return False
    return True


def circuit_breaker_get_status() -> dict:
    return {
        service: {
            "open": state["open"],
            "failures": state["failures"],
            "max_failures": _CIRCUIT_BREAKER_CONFIG[service]["max_failures"],
            "last_failure_ts": state["last_failure_ts"],
            "cooldown_remaining": max(
                0.0,
                _CIRCUIT_BREAKER_CONFIG[service]["cooldown_seconds"]
                - (_time_module.time() - state["last_failure_ts"]),
            ),
        }
        for service, state in _circuit_state.items()
    }
