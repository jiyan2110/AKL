"""Request-ID propagation, access log and HTTP metrics (PRD §8.2.2, Appendix F.1)."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from akl.api import metrics

log = logging.getLogger("akl.api.access")
_UUID4 = re.compile(r"^[0-9a-fA-F-]{32,36}$")


def route_template(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return str(path) if path else request.url.path


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _UUID4.match(incoming) else uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        route = request.url.path
        metrics.HTTP_INFLIGHT.labels(route=route).inc()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            elapsed = time.perf_counter() - start
            route = route_template(request)
            metrics.HTTP_INFLIGHT.labels(route=request.url.path).dec()
            metrics.HTTP_REQUESTS.labels(
                method=request.method, route=route, status=str(status)
            ).inc()
            metrics.HTTP_LATENCY.labels(method=request.method, route=route).observe(elapsed)
            log.info(
                "http_access",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "route": route,
                    "status": status,
                    "duration_ms": round(elapsed * 1000, 1),
                },
            )
        response.headers["X-Request-ID"] = request_id
        return response
