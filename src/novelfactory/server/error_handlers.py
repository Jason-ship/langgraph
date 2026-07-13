"""Standardized error handling for the NovelFactory FastAPI server.

Extracted from server/app.py (v6.1 P1-4) for single-responsibility separation.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def _error_response(
    status_code: int,
    code: str,
    message: str,
    details: dict | None = None,
) -> JSONResponse:
    """Build a standardized JSON error response."""
    error_id = str(uuid.uuid4())[:8]
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "error_id": error_id,
                "details": details or {},
            }
        },
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the FastAPI app.

    Call this after app creation to install:
      - ValueError → 400 INVALID_INPUT
      - FileNotFoundError → 404 NOT_FOUND
      - Exception → 500 INTERNAL_ERROR (with full traceback logging)
    """

    @app.exception_handler(ValueError)
    async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        return _error_response(400, "INVALID_INPUT", str(exc))

    @app.exception_handler(FileNotFoundError)
    async def not_found_handler(
        _request: Request, exc: FileNotFoundError
    ) -> JSONResponse:
        return _error_response(404, "NOT_FOUND", str(exc))

    @app.exception_handler(Exception)
    async def global_exception_handler(
        _request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error("[error] Unhandled exception: %s", exc, exc_info=True)
        return _error_response(500, "INTERNAL_ERROR", "An internal error occurred")
