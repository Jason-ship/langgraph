"""Rate limiting middleware for FastAPI — sliding window counter with Redis.

Supports configurable rate limits per endpoint prefix. When Redis is not
configured, automatically degrades to unlimited mode (no-op).

Usage:
    from novelfactory.middleware.rate_limit import RateLimitMiddleware

    app.add_middleware(
        RateLimitMiddleware,
        enabled=True,
        limits={
            "/api/runs": 60,
            "/api/threads": 120,
            "/api/store": 60,
        },
        window_seconds=60,
    )
"""

from __future__ import annotations

import logging
import time
from typing import Any

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

# ── Redis 键前缀 ────────────────────────────────────────────────────────────
_REDIS_KEY_PREFIX = "rate_limit:"
_REDIS_KEY_FORMAT = _REDIS_KEY_PREFIX + "{window}:{client_key}:{path_key}"

# ── 默认速率限制（每分钟请求数）───────────────────────────────────────────
_DEFAULT_LIMITS: dict[str, int] = {
    "/runs": 60,
    "/threads": 120,
    "/store": 60,
    "/assistants": 60,
}

# 免检路径前缀
_EXEMPT_PATHS: tuple[str, ...] = (
    "/health",
    "/ready",
    "/docs",
    "/redoc",
    "/openapi.json",
)

_RATE_LIMIT_REMAINING_HEADER = "X-RateLimit-Remaining"
_RATE_LIMIT_LIMIT_HEADER = "X-RateLimit-Limit"
_RATE_LIMIT_RESET_HEADER = "X-RateLimit-Reset"
_RETRY_AFTER_HEADER = "Retry-After"


class RateLimitMiddleware:
    """基于 Redis 滑动窗口计数器的速率限制中间件。

    对每个 (客户端IP, 路径前缀) 组合维护一个滑动窗口计数器。
    Redis 不可用时自动降级为无限制模式，不影响业务。

    窗口算法：
      - 每个请求到来时，清理窗口外的过期条目
      - 计算窗口内活跃请求数
      - 若超过限制返回 429
      - Redis key 的 TTL 为 window_seconds * 2，自动过期清理

    Attributes:
        app: ASGI 应用实例
        enabled: 是否启用限流
        limits: 路径前缀 → 每分钟最大请求数
        window_seconds: 滑动窗口时间窗口（秒）
        redis_client: 同步 redis.Redis 实例（可共享连接池）
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        enabled: bool = True,
        limits: dict[str, int] | None = None,
        window_seconds: int = 60,
    ) -> None:
        """初始化 RateLimitMiddleware。

        Args:
            app: ASGI 应用实例
            enabled: 是否启用限流
            limits: 路径前缀 → 请求数限制。默认为 _DEFAULT_LIMITS。
            window_seconds: 滑动窗口大小（秒）
        """
        self.app = app
        self.enabled = enabled
        self.limits = {**_DEFAULT_LIMITS, **(limits or {})}
        self.window_seconds = window_seconds

        # 尝试初始化 Redis 连接
        self._redis: Any = None
        self._redis_available = False
        self._init_redis()

    def _init_redis(self) -> None:
        """尝试初始化异步 Redis 连接。

        从 settings.REDIS_URL 创建 asyncio Redis 连接。失败时静默降级。
        """
        try:
            from novelfactory.config.settings import settings

            if not settings.REDIS_URL:
                logger.info(
                    "[rate_limit] REDIS_URL not configured — unlimited mode"
                )
                return

            import redis.asyncio as redis_async

            self._redis = redis_async.Redis.from_url(
                settings.REDIS_URL,
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=True,
            )
            self._redis_available = True
            logger.info("[rate_limit] Redis connected — rate limiting enabled")
        except Exception:
            self._redis = None
            self._redis_available = False
            logger.warning(
                "[rate_limit] Redis unavailable — degraded to unlimited mode"
            )

    def _get_client_key(self, scope: Scope) -> str:
        """从请求 scope 提取客户端标识（IP 地址）。

        优先使用 X-Forwarded-For，回退到 remote_addr。

        Args:
            scope: ASGI scope 字典

        Returns:
            客户端标识字符串
        """
        headers = dict(scope.get("headers", []))
        forwarded = headers.get(b"x-forwarded-for")
        if forwarded:
            return forwarded.decode("utf-8", errors="replace").split(",")[0].strip()
        client = scope.get("client")
        if client:
            return client[0]
        return "unknown"

    def _match_path(self, path: str) -> tuple[str, int] | None:
        """匹配请求路径对应的速率限制配置。

        Args:
            path: 请求路径

        Returns:
            (匹配前缀, 限制数) 元组；未匹配返回 None
        """
        # 免检路径
        for exempt in _EXEMPT_PATHS:
            if path.startswith(exempt):
                return None

        # 按最长前缀优先匹配
        matched_prefix = ""
        matched_limit = 0
        for prefix, limit in self.limits.items():
            if path.startswith(prefix) and len(prefix) > len(matched_prefix):
                matched_prefix = prefix
                matched_limit = limit

        if matched_prefix:
            return (matched_prefix, matched_limit)
        return None

    async def _check_rate_limit(self, client_key: str, path_key: str, limit: int) -> dict[str, Any]:
        """检查并记录速率限制（滑动窗口计数器，async）。

        使用 Redis sorted set 实现滑动窗口：
          - member: 当前时间戳（毫秒）
          - score: 当前时间戳（毫秒）
          - 窗口外成员自动被移除
          - 原子操作确保并发安全

        Args:
            client_key: 客户端标识
            path_key: 路径标识
            limit: 窗口内最大请求数

        Returns:
            {
                "allowed": bool,
                "remaining": int,
                "reset_after": float,  # 秒
            }
        """
        if not self._redis_available or self._redis is None:
            return {"allowed": True, "remaining": limit, "reset_after": 0.0}

        now = time.time()
        window_start = now - self.window_seconds
        window_key = _REDIS_KEY_FORMAT.format(
            window=int(now // self.window_seconds),
            client_key=client_key,
            path_key=path_key,
        )

        try:
            pipeline = self._redis.pipeline()  # type: ignore[union-attr]

            # 移除窗口外的旧记录
            pipeline.zremrangebyscore(window_key, 0, window_start * 1000)

            # 添加当前请求（毫秒精度）
            pipeline.zadd(window_key, {str(now * 1000): now * 1000})

            # 统计窗口内请求数
            pipeline.zcard(window_key)

            # 设置 TTL
            pipeline.expire(window_key, self.window_seconds * 2)

            _, _, count, _ = await pipeline.execute()  # type: ignore[union-attr]

            remaining = max(0, limit - int(count))
            allowed = remaining > 0

            return {
                "allowed": allowed,
                "remaining": remaining,
                "reset_after": float(self.window_seconds),
            }
        except Exception:
            logger.debug("[rate_limit] Redis check failed — allowing request")
            return {"allowed": True, "remaining": limit, "reset_after": 0.0}

    async def _send_429(
        self,
        send: Send,
        reset_after: float,
    ) -> None:
        """发送 429 Too Many Requests 响应。

        Args:
            send: ASGI send 回调
            reset_after: 重试等待秒数
        """
        body = (
            b'{"error":"Too Many Requests",'
            b'"detail":"Rate limit exceeded. Please try again later."}'
        )
        headers = [
            (b"content-type", b"application/json"),
            (_RETRY_AFTER_HEADER.encode(), str(int(reset_after)).encode()),
            (b"content-length", str(len(body)).encode()),
        ]

        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": headers,
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
            }
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.enabled or not self._redis_available:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        match = self._match_path(path)

        # 未匹配到限流规则 → 直接通过
        if match is None:
            await self.app(scope, receive, send)
            return

        path_key, limit = match
        client_key = self._get_client_key(scope)
        result = await self._check_rate_limit(client_key, path_key, limit)

        if not result["allowed"]:
            logger.warning(
                "[rate_limit] 429 client=%s path=%s remaining=%d",
                client_key,
                path,
                result["remaining"],
            )
            await self._send_429(send, result["reset_after"])
            return

        # 在响应头中注入速率限制信息
        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers[_RATE_LIMIT_LIMIT_HEADER] = str(limit)
                headers[_RATE_LIMIT_REMAINING_HEADER] = str(result["remaining"])
                headers[_RATE_LIMIT_RESET_HEADER] = str(int(time.time() + result["reset_after"]))
            await send(message)

        await self.app(scope, receive, send_with_headers)


__all__ = [
    "RateLimitMiddleware",
]
