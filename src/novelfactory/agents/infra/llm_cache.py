"""LLM 响应缓存 — 基于 Redis 的语义缓存。

v5.4: 新模块。缓存 LLM 响应用于减少重复 API 调用。
缓存键: sha256(prompt[:2000]) + model + temperature
TTL: 默认 3600s (1小时)，写作场景下同章重试可能命中。

优雅降级: Redis 不可用时跳过缓存，不影响主流程。
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ── 默认配置 ────────────────────────────────────────────────────────────────
_DEFAULT_TTL = 3600  # 1 小时
_CACHE_KEY_PREFIX = "llm_cache:"
_PROMPT_HASH_LENGTH = 2000  # 取 prompt 前 2000 字符做 hash


def _build_cache_key(model: str, temperature: float, prompt: str) -> str:
    """构建缓存键: llm_cache:{model}:{temp}:{prompt_hash}"""
    prompt_hash = hashlib.sha256(prompt[:_PROMPT_HASH_LENGTH].encode()).hexdigest()[:16]
    return f"{_CACHE_KEY_PREFIX}{model}:t{temperature:.2f}:{prompt_hash}"


class LLMResponseCache:
    """基于 Redis 的 LLM 响应缓存。

    用法:
        cache = LLMResponseCache("redis://localhost:6379")
        cached = await cache.get("deepseek-v4-flash", 0.7, prompt)
        if cached is None:
            response = await llm.ainvoke(...)
            await cache.set("deepseek-v4-flash", 0.7, prompt, response)
    """

    def __init__(
        self,
        redis_url: str | None = None,
        default_ttl: int = _DEFAULT_TTL,
    ) -> None:
        self._default_ttl = default_ttl
        self._redis: Any = None
        self._redis_available = False
        # v5.5-fix: cache hit/miss tracking
        self._hits: int = 0
        self._misses: int = 0

        # v6.1: 统一从 settings 读取
        from novelfactory.config.settings import settings as _st

        redis_url = (
            redis_url
            or _st.redis_url
            or os.environ.get("REDIS_URL")
            or os.environ.get("REDIS_CONNECTION_STRING")
        )

        if redis_url:
            try:
                import redis.asyncio as aioredis

                self._redis = aioredis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=3,
                    socket_timeout=3,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )
                self._redis_available = True
                logger.info("[llm_cache] Redis connected: %s", redis_url[:40])
            except Exception as e:
                logger.warning("[llm_cache] Redis init failed (%s), cache disabled", e)
                self._redis_available = False
        else:
            logger.info("[llm_cache] No Redis URL configured, cache disabled")
            self._redis_available = False

    async def get(
        self,
        model: str,
        temperature: float,
        prompt: str,
    ) -> str | None:
        """从缓存获取 LLM 响应。

        Returns:
            缓存的响应字符串，未命中或 Redis 不可用时返回 None。
        """
        if not self._redis_available or not self._redis:
            self._misses += 1  # cache unavailable = miss
            return None

        try:
            key = _build_cache_key(model, temperature, prompt)
            value = await self._redis.get(key)
            if value:
                self._hits += 1
                logger.debug("[llm_cache] HIT: model=%s t=%.2f", model, temperature)
            else:
                self._misses += 1
            return value
        except Exception as e:
            self._misses += 1
            logger.debug("[llm_cache] get failed: %s", e)
            return None

    async def set(
        self,
        model: str,
        temperature: float,
        prompt: str,
        response: str,
    ) -> None:
        """写入 LLM 响应到缓存。"""
        if not self._redis_available or not self._redis:
            return

        try:
            key = _build_cache_key(model, temperature, prompt)
            await self._redis.setex(key, self._default_ttl, response)
        except Exception as e:
            logger.debug("[llm_cache] set failed: %s", e)

    @property
    def available(self) -> bool:
        """缓存是否可用。"""
        return self._redis_available

    @property
    def hits(self) -> int:
        """缓存命中次数。"""
        return self._hits

    @property
    def misses(self) -> int:
        """缓存未命中次数。"""
        return self._misses

    @property
    def hit_rate(self) -> float:
        """缓存命中率 (0.0 - 1.0)。"""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total


# ── 模块级单例 ────────────────────────────────────────────────────────────────

_cache_instance: LLMResponseCache | None = None


def get_llm_cache() -> LLMResponseCache:
    """获取模块级 LLMResponseCache 单例。"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = LLMResponseCache()
    return _cache_instance
