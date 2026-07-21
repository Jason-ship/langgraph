"""CSRF 中间件 — Double Submit Cookie 模式。

参考 DeerFlow app/gateway/csrf_middleware.py。

对 POST/PUT/DELETE/PATCH 请求检查 CSRF Token。
"""

from __future__ import annotations

import logging
import secrets
from typing import Any

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

CSRF_COOKIE_NAME = "csrf_token"
CSRF_HEADER_NAME = "X-CSRF-Token"

# 免检路径 — 仅豁免只读公开端点和使用自有签名验证的 Webhook。
# SDK 端点（/threads、/runs 等）不豁免，需携带 CSRF Token。
_EXEMPT_PATHS = frozenset({
    "/health",
    "/ready",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/metrics",
    "/favicon.ico",
    "/api/webhooks/",
})


def generate_csrf_token() -> str:
    """生成 CSRF Token。"""
    return secrets.token_urlsafe(32)


class CSRFMiddleware:
    """Double Submit Cookie 模式 CSRF 保护中间件。

    豁免路径：
    - GET/HEAD/OPTIONS 请求
    - /health, /ready, /docs 等公开路径
    - /api/webhooks/ 开头的 Webhook 路径（使用自有签名验证）
    """

    def __init__(self, app: ASGIApp, *, enabled: bool = True):
        self.app = app
        self.enabled = enabled

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.enabled:
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET")
        path = scope.get("path", "/")

        # GET/HEAD/OPTIONS 和豁免路径不检查
        if method in ("GET", "HEAD", "OPTIONS"):
            await self.app(scope, receive, send)
            return

        if any(path.startswith(exempt) for exempt in _EXEMPT_PATHS):
            await self.app(scope, receive, send)
            return

        # 检查 CSRF Token
        from starlette.datastructures import Headers

        headers = Headers(scope=scope)
        cookie_token = ""
        for cookie in headers.get("cookie", "").split(";"):
            cookie = cookie.strip()
            if cookie.startswith(f"{CSRF_COOKIE_NAME}="):
                cookie_token = cookie[len(f"{CSRF_COOKIE_NAME}="):]
                break

        header_token = headers.get(CSRF_HEADER_NAME, "")

        if not cookie_token or not secrets.compare_digest(cookie_token, header_token):
            logger.warning("[csrf] Invalid CSRF token: method=%s path=%s", method, path)
            response_headers = MutableHeaders()
            response_headers["Content-Type"] = "application/json"
            response_headers["X-CSRF-Failed"] = "1"

            async def send_error(message: Message) -> None:
                if message["type"] == "http.response.start":
                    message["status"] = 403
                    for k, v in response_headers.items():
                        MutableHeaders(scope=message)[k] = v
                elif message["type"] == "http.response.body":
                    message["body"] = b'{"detail":"CSRF validation failed"}'
                await send(message)

            await self.app(scope, receive, send_error)
            return

        await self.app(scope, receive, send)


__all__ = [
    "CSRFMiddleware",
    "generate_csrf_token",
    "CSRF_COOKIE_NAME",
    "CSRF_HEADER_NAME",
]