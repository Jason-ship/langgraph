"""Redis 独立 store 层 — 通用应用数据缓存与持久化。

提供异步 Redis 操作，覆盖：
  - 键值存储（get/set/delete，支持 TTL）
  - Hash 操作（hget/hset/hgetall）
  - List 操作（lpush/lrange/ltrim）
  - 存在性检查（exists）
  - 优雅降级（Redis 不可用时自动跳过，不影响主流程）

设计原则：
  - 线程安全的模块级单例（get_redis_store）
  - 与 LLMResponseCache 共享同一 Redis 连接池
  - 所有方法均为 async，适配 LangGraph async 运行环境
  - 失败时返回安全默认值（None/[]/False），不抛异常

Usage:
    from novelfactory.store.redis_store import get_redis_store

    store = get_redis_store()
    await store.set("key", "value", ttl=3600)
    value = await store.get("key")
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

# ── Module-level singleton ──────────────────────────────────────────────────────
_redis_store: RedisStore | None = None
_store_lock = threading.Lock()


def get_redis_store() -> RedisStore:
    """Get or create the module-level RedisStore singleton (thread-safe)."""
    global _redis_store
    if _redis_store is None:
        with _store_lock:
            if _redis_store is None:
                _redis_store = RedisStore()
    return _redis_store


# ═══════════════════════════════════════════════════════════════════════════════
# RedisStore
# ═══════════════════════════════════════════════════════════════════════════════


class RedisStore:
    """Async Redis store for general-purpose application data.

    Auto-detects Redis connection from environment variables:
      REDIS_URL > REDIS_HOST+REDIS_PORT+REDIS_DB+REDIS_PASSWORD.

    Graceful degradation: if Redis is unavailable, all read operations
    return safe defaults (None/[]/False) and writes are silently skipped.
    """

    def __init__(self, redis_url: str | None = None) -> None:
        self._redis: Any = None
        self._available = False

        url = redis_url or self._build_url()
        if not url:
            logger.info("RedisStore: no Redis URL configured, store unavailable")
            return

        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                health_check_interval=30,
            )
            self._available = True
            logger.info("RedisStore connected: %s", self._mask_url(url))
        except Exception as e:
            logger.warning("RedisStore init failed (graceful degradation): %s", e)

    @staticmethod
    def _build_url() -> str | None:
        """Build Redis URL from environment variables."""
        # v6.1: 统一从 settings 读取
        from novelfactory.config.settings import settings as _st

        url = os.environ.get("REDIS_URL") or os.environ.get("REDIS_CONNECTION_STRING")
        if url:
            return url
        host = _st.REDIS_HOST or os.environ.get("REDIS_HOST", "localhost")
        port = (
            str(_st.REDIS_PORT)
            if _st.REDIS_PORT
            else os.environ.get("REDIS_PORT", "6379")
        )
        db = str(_st.REDIS_DB) if _st.REDIS_DB else os.environ.get("REDIS_DB", "0")
        password = _st.REDIS_PASSWORD or os.environ.get("REDIS_PASSWORD", "")
        if password:
            from urllib.parse import quote

            return f"redis://:{quote(password)}@{host}:{port}/{db}"
        return f"redis://{host}:{port}/{db}"

    @staticmethod
    def _mask_url(url: str) -> str:
        """Mask password in Redis URL for safe logging."""
        import re

        return re.sub(r"://:([^@]+)@", "://:***@", url)

    @property
    def available(self) -> bool:
        """Whether Redis is connected and ready."""
        return self._available

    def is_connected(self) -> bool:
        """Whether Redis is connected and ready (Protocol-compliant alias for ``available``)."""
        return self._available

    async def _ensure_connected(self) -> bool:
        """Lazy connection check with auto-reconnect. Returns True if Redis is available.

        v5.9 FIX: Redis 闪断后自动尝试重连，而非永久降级。
        """
        if self._available:
            return True
        if self._redis is None:
            return False
        # v5.9: 尝试重连 — ping 失败后重建连接
        try:
            await self._redis.ping()
            self._available = True
            return True
        except Exception:
            # 尝试重建连接
            try:
                await self._redis.aclose()
            except Exception:
                pass
            url = self._build_url()
            if not url:
                self._available = False
                return False
            try:
                import redis.asyncio as aioredis

                self._redis = aioredis.from_url(
                    url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    health_check_interval=30,
                )
                await self._redis.ping()
                self._available = True
                logger.info("RedisStore reconnected: %s", self._mask_url(url))
                return True
            except Exception as e:
                logger.warning("RedisStore reconnect failed: %s", e)
                self._available = False
                return False

    # ── Key-Value Operations ─────────────────────────────────────────────────

    async def get(self, key: str) -> str | None:
        """Get a string value by key. Returns None if key not found or Redis unavailable."""
        if not await self._ensure_connected():
            return None
        try:
            return await self._redis.get(key)
        except Exception as e:
            logger.warning("RedisStore get(%s) failed: %s", key, e)
            return None

    async def set(self, key: str, value: str, ttl: int | None = None) -> bool:
        """Set a string value with optional TTL (seconds)."""
        if not await self._ensure_connected():
            return False
        try:
            if ttl:
                await self._redis.setex(key, ttl, value)
            else:
                await self._redis.set(key, value)
            return True
        except Exception as e:
            logger.warning("RedisStore set(%s) failed: %s", key, e)
            return False

    async def delete(self, *keys: str) -> int:
        """Delete one or more keys. Returns number of keys deleted."""
        if not keys or not await self._ensure_connected():
            return 0
        try:
            return await self._redis.delete(*keys)
        except Exception as e:
            logger.warning("RedisStore delete failed: %s", e)
            return 0

    async def exists(self, *keys: str) -> int:
        """Check if keys exist. Returns count of existing keys."""
        if not keys or not await self._ensure_connected():
            return 0
        try:
            return await self._redis.exists(*keys)
        except Exception as e:
            logger.warning("RedisStore exists failed: %s", e)
            return 0

    async def get_json(self, key: str) -> dict | list | None:
        """Get a JSON-deserialized value."""
        raw = await self.get(key)
        if raw is None:
            return None
        try:
            import json

            return json.loads(raw)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("RedisStore get_json(%s) parse failed: %s", key, e)
            return None

    async def set_json(
        self, key: str, value: dict | list, ttl: int | None = None
    ) -> bool:
        """Set a JSON-serialized value."""
        import json

        return await self.set(key, json.dumps(value, ensure_ascii=False), ttl=ttl)

    # ── Hash Operations ──────────────────────────────────────────────────────

    async def hget(self, name: str, key: str) -> str | None:
        """Get a field value from a hash."""
        if not await self._ensure_connected():
            return None
        try:
            return await self._redis.hget(name, key)
        except Exception as e:
            logger.warning("RedisStore hget(%s, %s) failed: %s", name, key, e)
            return None

    async def hset(self, name: str, key: str, value: str) -> bool:
        """Set a field in a hash."""
        if not await self._ensure_connected():
            return False
        try:
            await self._redis.hset(name, key, value)
            return True
        except Exception as e:
            logger.warning("RedisStore hset(%s, %s) failed: %s", name, key, e)
            return False

    async def hgetall(self, name: str) -> dict[str, str]:
        """Get all fields and values from a hash."""
        if not await self._ensure_connected():
            return {}
        try:
            return await self._redis.hgetall(name)
        except Exception as e:
            logger.warning("RedisStore hgetall(%s) failed: %s", name, e)
            return {}

    async def hdel(self, name: str, *keys: str) -> int:
        """Delete fields from a hash."""
        if not keys or not await self._ensure_connected():
            return 0
        try:
            return await self._redis.hdel(name, *keys)
        except Exception as e:
            logger.warning("RedisStore hdel(%s) failed: %s", name, e)
            return 0

    # ── List Operations ──────────────────────────────────────────────────────

    async def lpush(self, name: str, *values: str) -> int:
        """Prepend values to a list. Returns list length after push."""
        if not values or not await self._ensure_connected():
            return 0
        try:
            return await self._redis.lpush(name, *values)
        except Exception as e:
            logger.warning("RedisStore lpush(%s) failed: %s", name, e)
            return 0

    async def lrange(self, name: str, start: int, end: int) -> list[str]:
        """Get a range of elements from a list."""
        if not await self._ensure_connected():
            return []
        try:
            return await self._redis.lrange(name, start, end)
        except Exception as e:
            logger.warning("RedisStore lrange(%s) failed: %s", name, e)
            return []

    async def ltrim(self, name: str, start: int, end: int) -> bool:
        """Trim a list to the specified range."""
        if not await self._ensure_connected():
            return False
        try:
            await self._redis.ltrim(name, start, end)
            return True
        except Exception as e:
            logger.warning("RedisStore ltrim(%s) failed: %s", name, e)
            return False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis:
            try:
                await self._redis.aclose()
            except Exception as e:
                logger.warning("RedisStore close error: %s", e)
            finally:
                self._available = False

    def close_sync(self) -> None:
        """Synchronous close wrapper (for cleanup in non-async contexts)."""
        if self._redis:
            try:
                self._redis.close()
            except Exception:
                pass
            finally:
                self._available = False
