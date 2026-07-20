"""FastAPI request logging middleware — logs method, path, status, duration, and user-agent.

Usage:
    from novelfactory.middleware.request_logger import RequestLoggingMiddleware

    app.add_middleware(RequestLoggingMiddleware)

Sensitive paths (/health, /ready, /metrics) are automatically skipped.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# ── Paths excluded from logging ────────────────────────────────────────────
_SKIP_PATHS: frozenset[str] = frozenset({"/health", "/ready", "/metrics"})


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with method, path, status code, duration, and user-agent.

    - Production (settings.is_production is True): logs at INFO level.
    - Development / staging: logs at DEBUG level.
    - Paths in ``_SKIP_PATHS`` are silently skipped.
    - Duration is measured via ``time.perf_counter()`` with millisecond precision.

    Compatible with ``app.add_middleware()``:

        app.add_middleware(RequestLoggingMiddleware)
    """

    def __init__(self, app: Any, *, skip_paths: frozenset[str] | None = None) -> None:
        """Initialise middleware.

        Args:
            app: The ASGI application to wrap.
            skip_paths: Override the default set of skipped paths. If ``None``,
                uses ``{"/health", "/ready", "/metrics"}``.
        """
        super().__init__(app)
        self._skip_paths = _SKIP_PATHS if skip_paths is None else skip_paths

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Dispatch and log the request.

        Args:
            request: Incoming HTTP request.
            call_next: Callable that processes the request and returns a response.

        Returns:
            The HTTP response from the downstream application.
        """
        path = request.url.path

        # ── Skip sensitive / health-check paths ──────────────────────────
        if path in self._skip_paths:
            return await call_next(request)

        method = request.method
        user_agent = request.headers.get("user-agent", "-")

        start = time.perf_counter()
        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start) * 1000
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            # Log the exception-outcome request before re-raising
            self._log(method=method, path=path, status_code=0, duration_ms=duration_ms, user_agent=user_agent)
            raise

        self._log(method=method, path=path, status_code=response.status_code, duration_ms=duration_ms, user_agent=user_agent)
        return response

    def _log(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        user_agent: str,
    ) -> None:
        """Emit the log record at the appropriate level.

        Production environments use INFO; all others use DEBUG.
        """
        # Lazy import to avoid circular dependencies at module level
        from novelfactory.config.settings import settings

        log_level = logger.info if settings.is_production else logger.debug
        log_level(
            "[request] %s %s → %s (%.1f ms) ua=%s",
            method,
            path,
            status_code,
            duration_ms,
            user_agent,
        )


__all__ = [
    "RequestLoggingMiddleware",
]
