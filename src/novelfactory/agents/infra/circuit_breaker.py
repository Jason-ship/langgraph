"""Per-service circuit breaker for LLM providers (ARK/DeepSeek/硅基流动).

v7.8: 新增 RedisCircuitBreakerStore，支持 Redis 持久化熔断状态。
       保留现有内存 _circuit_state 作为 fallback。
"""

from __future__ import annotations

import json
import time as _time_module
from typing import Any

from novelfactory.agents.infra.logger import get_logger
from novelfactory.config.constants import (
    CIRCUIT_BREAKER_CONFIG as _CIRCUIT_BREAKER_CONFIG,
)

# v5.1.1: 完善熔断器，覆盖实际使用的 LLM provider (ARK/DeepSeek/硅基流动)。
# "matrix" 保留为兼容历史配置。
# v6.1: 配置统一从 config.constants 导入（唯一真实来源）。

_logger = get_logger("novelfactory.circuit_breaker")

# ── 内存熔断状态（fallback）───────────────────────────────────────────────
_circuit_state: dict[str, dict[str, Any]] = {
    name: {"failures": 0, "last_failure_ts": 0.0, "open": False}
    for name in _CIRCUIT_BREAKER_CONFIG
}

# ── Redis 存储键前缀 ──────────────────────────────────────────────────────
_REDIS_KEY_PREFIX = "circuit_breaker:"
_REDIS_KEY_SERVICE = _REDIS_KEY_PREFIX + "service:{}"
_REDIS_KEY_ALL_SERVICES = _REDIS_KEY_PREFIX + "services"


class RedisCircuitBreakerStore:
    """Redis 持久化熔断状态存储。

    通过同步 redis.Redis 实例读写熔断状态。
    当 Redis 不可用或未配置时静默降级，回退到内存状态。
    """

    def __init__(self, redis_client: redis.Redis | None = None) -> None:  # type: ignore[name-defined]  # noqa: F821
        """初始化 Redis 存储。

        Args:
            redis_client: 同步 redis.Redis 实例。为 None 时尝试从 settings 创建。
        """
        self._redis: redis.Redis | None = redis_client  # type: ignore[name-defined]  # noqa: F821
        self._enabled = False

        if self._redis is None:
            self._init_from_settings()
        elif self._ping():
            self._enabled = True

    def _init_from_settings(self) -> None:
        """尝试从 settings.REDIS_URL 初始化 Redis 连接。"""
        try:
            from novelfactory.config.settings import settings

            if not settings.REDIS_URL:
                return
            import redis as _redis_lib

            self._redis = _redis_lib.Redis.from_url(
                settings.REDIS_URL,
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=True,
            )
            if self._ping():
                self._enabled = True
                _logger.info(
                    "[circuit_breaker] Redis store enabled (url=%s)",
                    _mask_redis_url(settings.REDIS_URL),
                )
        except Exception:
            self._redis = None
            self._enabled = False
            _logger.debug("[circuit_breaker] Redis unavailable, using in-memory store")

    def _ping(self) -> bool:
        """检查 Redis 连接是否可用。"""
        try:
            return bool(self._redis and self._redis.ping())
        except Exception:
            return False

    @property
    def enabled(self) -> bool:
        """Redis 存储是否可用。"""
        return self._enabled

    def get_state(self, service: str) -> dict[str, Any] | None:
        """从 Redis 读取指定服务的熔断状态。

        Args:
            service: 服务名称 (e.g. "ark", "deepseek")

        Returns:
            熔断状态字典，或 None（不存在或 Redis 不可用）
        """
        if not self._enabled or not self._redis:
            return None
        try:
            raw = self._redis.get(_REDIS_KEY_SERVICE.format(service))
            if raw is None:
                return None
            state: dict[str, Any] = json.loads(raw)
            # 合并缺失字段（兼容旧格式）
            state.setdefault("failures", 0)
            state.setdefault("last_failure_ts", 0.0)
            state.setdefault("open", False)
            return state
        except Exception:
            return None

    def set_state(self, service: str, state: dict[str, Any]) -> bool:
        """将熔断状态写入 Redis。

        Args:
            service: 服务名称
            state: 熔断状态字典

        Returns:
            是否写入成功
        """
        if not self._enabled or not self._redis:
            return False
        try:
            key = _REDIS_KEY_SERVICE.format(service)
            self._redis.set(key, json.dumps(state))
            self._redis.sadd(_REDIS_KEY_ALL_SERVICES, service)
            # 设置 TTL = cooldown_seconds * 3，防止孤立数据堆积
            ttl = int(
                _CIRCUIT_BREAKER_CONFIG.get(service, {}).get("cooldown_seconds", 30)
                * 3
            )
            self._redis.expire(key, ttl)
            return True
        except Exception:
            return False

    def get_all_states(self) -> dict[str, dict[str, Any]]:
        """读取 Redis 中所有服务的熔断状态。

        Returns:
            服务名 → 熔断状态字典
        """
        if not self._enabled or not self._redis:
            return {}
        try:
            service_names = self._redis.smembers(_REDIS_KEY_ALL_SERVICES)
            result: dict[str, dict[str, Any]] = {}
            for name in service_names:
                state = self.get_state(name)
                if state is not None:
                    result[name] = state
            return result
        except Exception:
            return {}

    def reset(self, service: str | None = None) -> None:
        """清除指定服务或所有服务的 Redis 熔断状态。

        Args:
            service: 服务名称。为 None 时清除所有。
        """
        if not self._enabled or not self._redis:
            return
        try:
            if service:
                self._redis.delete(_REDIS_KEY_SERVICE.format(service))
                self._redis.srem(_REDIS_KEY_ALL_SERVICES, service)
            else:
                for name in list(_CIRCUIT_BREAKER_CONFIG):
                    self._redis.delete(_REDIS_KEY_SERVICE.format(name))
                self._redis.delete(_REDIS_KEY_ALL_SERVICES)
        except Exception:
            pass


def _mask_redis_url(url: str) -> str:
    """掩码 Redis URL 中的密码部分，用于日志输出。"""
    if "@" in url:
        scheme, rest = url.split("://", 1)
        userinfo, hostpart = rest.split("@", 1)
        if ":" in userinfo:
            user, _ = userinfo.split(":", 1)
            return f"{scheme}://{user}:****@{hostpart}"
        return f"{scheme}://{userinfo}:****@{hostpart}"
    return url


# ── 全局 Redis 存储实例（懒初始化）─────────────────────────────────────────
_redis_store: RedisCircuitBreakerStore | None = None


def _get_redis_store() -> RedisCircuitBreakerStore | None:
    """获取全局 RedisCircuitBreakerStore 实例（懒初始化）。"""
    global _redis_store
    if _redis_store is None:
        _redis_store = RedisCircuitBreakerStore()
    if _redis_store.enabled:
        return _redis_store
    return None


def _sync_redis_to_memory(service: str) -> None:
    """将 Redis 状态同步到内存 _circuit_state。

    用于模块启动或 Redis 恢复后同步到内存，确保读取一致性。
    """
    store = _get_redis_store()
    if store is None or service not in _circuit_state:
        return
    redis_state = store.get_state(service)
    if redis_state is not None:
        _circuit_state[service].update(redis_state)


def circuit_breaker_record_success(service: str) -> None:
    """记录一次服务调用成功，重置熔断状态。

    Args:
        service: 服务名称 (e.g. "ark", "deepseek")
    """
    if service in _circuit_state:
        _circuit_state[service]["failures"] = 0
        _circuit_state[service]["open"] = False
    # 同步写入 Redis
    store = _get_redis_store()
    if store is not None and service in _CIRCUIT_BREAKER_CONFIG:
        store.set_state(service, dict(_circuit_state.get(service, {})))


def circuit_breaker_record_failure(service: str) -> None:
    """记录一次服务调用失败，达到阈值时打开熔断器。

    Args:
        service: 服务名称 (e.g. "ark", "deepseek")
    """
    if service not in _circuit_state or service not in _CIRCUIT_BREAKER_CONFIG:
        return
    state = _circuit_state[service]
    cfg = _CIRCUIT_BREAKER_CONFIG[service]
    state["failures"] += 1
    state["last_failure_ts"] = _time_module.time()
    if state["failures"] >= cfg["max_failures"]:
        state["open"] = True
        _logger.warning(
            "[circuit_breaker] service='%s' OPEN after %d failures — fast-failing for %ds",
            service,
            state["failures"],
            cfg["cooldown_seconds"],
        )
    # 同步写入 Redis
    store = _get_redis_store()
    if store is not None:
        store.set_state(service, dict(state))


def circuit_breaker_is_open(service: str) -> bool:
    """检查指定服务的熔断器是否打开（处于快速失败状态）。

    支持 Redis 持久化：优先从 Redis 读取状态，Redis 不可用时回退到内存。

    Args:
        service: 服务名称

    Returns:
        True 表示熔断器打开，应快速失败
    """
    if service not in _circuit_state or service not in _CIRCUIT_BREAKER_CONFIG:
        return False

    # 尝试从 Redis 同步最新状态
    store = _get_redis_store()
    if store is not None:
        redis_state = store.get_state(service)
        if redis_state is not None:
            _circuit_state[service].update(redis_state)

    state = _circuit_state[service]
    cfg = _CIRCUIT_BREAKER_CONFIG[service]
    if not state["open"]:
        return False
    elapsed = _time_module.time() - state["last_failure_ts"]
    if elapsed >= cfg["cooldown_seconds"]:
        state["open"] = False
        state["failures"] = 0
        _logger.info(
            "[circuit_breaker] service='%s' cooldown expired — probing",
            service,
        )
        if store is not None:
            store.set_state(service, dict(state))
        return False
    return True


def circuit_breaker_get_status() -> dict[str, dict[str, Any]]:
    """获取所有服务熔断器状态摘要。

    合并内存和 Redis 中的状态，Redis 状态优先。

    Returns:
        服务名 → 状态字典（含 open/failures/max_failures/cooldown_remaining 等）
    """
    result: dict[str, dict[str, Any]] = {}

    # 先从内存获取
    for service, state in _circuit_state.items():
        if service in _CIRCUIT_BREAKER_CONFIG:
            result[service] = {
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

    # 从 Redis 补充/覆盖
    store = _get_redis_store()
    if store is not None:
        redis_states = store.get_all_states()
        for service, redis_state in redis_states.items():
            if service in _CIRCUIT_BREAKER_CONFIG:
                cfg = _CIRCUIT_BREAKER_CONFIG[service]
                result[service] = {
                    "open": redis_state["open"],
                    "failures": redis_state["failures"],
                    "max_failures": cfg["max_failures"],
                    "last_failure_ts": redis_state["last_failure_ts"],
                    "cooldown_remaining": max(
                        0.0,
                        cfg["cooldown_seconds"]
                        - (_time_module.time() - redis_state["last_failure_ts"]),
                    ),
                }

    return result
