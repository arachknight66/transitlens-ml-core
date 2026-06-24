"""
api/middleware.py
-----------------
FastAPI middleware for request logging, timing, and error handling.

Middleware stack (order matters — outermost runs first):
    1. TimingMiddleware     — records X-Processing-Time-Ms header
    2. ErrorHandlerMiddleware — catches unhandled exceptions → 500 JSON
    3. Logging              — logs method, path, status, and duration

Used by: api/app.py
"""

from __future__ import annotations

import logging
import time as _time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger("api")


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Add ``X-Processing-Time-Ms`` header to every response.

    Also logs each request's method, path, status code, and processing time.
    Format: ``POST /analyze 200 1234ms``
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start = _time.perf_counter()

        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed_ms = (_time.perf_counter() - start) * 1000
            logger.error(
                "%s %s 500 %.0fms (unhandled: %s)",
                request.method, request.url.path, elapsed_ms, exc,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "detail": f"Internal server error: {type(exc).__name__}: {exc}",
                },
            )

        elapsed_ms = (_time.perf_counter() - start) * 1000
        response.headers["X-Processing-Time-Ms"] = f"{elapsed_ms:.2f}"

        logger.info(
            "%s %s %s %.0fms",
            request.method, request.url.path, response.status_code, elapsed_ms,
        )

        return response
