"""Request trace middleware — binds trace ID to HTTP requests.

Migrated from DeerFlow app/gateway/trace_middleware.py.
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

TRACE_ID_HEADER = "X-Trace-Id"
_trace_id_ctx: ContextVar[str] = ContextVar("trace_id", default="")


def get_trace_id() -> str:
    """Get the current request trace ID."""
    return _trace_id_ctx.get()


class TraceMiddleware:
    """Bind a request-level trace id and write it to HTTP response headers."""

    def __init__(self, app: ASGIApp, *, enabled: bool = True):
        self.app = app
        self.enabled = enabled

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.enabled:
            await self.app(scope, receive, send)
            return

        trace_id = str(uuid.uuid4())
        token = _trace_id_ctx.set(trace_id)

        try:
            async def send_with_trace(message: Message) -> None:
                if message["type"] == "http.response.start":
                    from starlette.datastructures import MutableHeaders

                    headers = MutableHeaders(scope=message)
                    headers[TRACE_ID_HEADER] = trace_id
                await send(message)

            logger.debug("[trace] Request bound trace_id=%s path=%s", trace_id, scope.get("path", "?"))
            await self.app(scope, receive, send_with_trace)
        finally:
            _trace_id_ctx.reset(token)


__all__ = [
    "TraceMiddleware",
    "get_trace_id",
    "TRACE_ID_HEADER",
]