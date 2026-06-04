"""FastAPI middleware for request tracing and timing.

Provides a pure-ASGI middleware that injects X-Request-ID and measures
round-trip response time, appending both as response headers.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that adds X-Request-ID and X-Response-Time-ms headers.

    If the incoming request already carries an X-Request-ID header (e.g.,
    from an upstream load balancer or API gateway), that value is preserved
    for distributed tracing. Otherwise, a new UUIDv4 is generated.

    The X-Response-Time-ms header records the wall-clock time (in
    milliseconds) the server spent processing the request, measured from
    when this middleware receives the request until the response is ready.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # Propagate or generate request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())

        # Attach to request state so downstream handlers can read it
        request.state.request_id = request_id

        start_time = time.perf_counter()

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.2f}"

        return response
