"""Request middleware — request IDs, access logging, latency metrics."""
from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from ingestion.logging_config import get_logger

log = get_logger("api.access")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request ID, times the request, logs it, feeds the monitor."""

    def __init__(self, app, monitor):
        super().__init__(app)
        self.monitor = monitor

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            latency_ms = (time.perf_counter() - t0) * 1000
            self.monitor.record_request(latency_ms, is_error=True)
            log.error("rid=%s %s %s -> 500 (%.2f ms)", request_id,
                      request.method, request.url.path, latency_ms)
            raise
        latency_ms = (time.perf_counter() - t0) * 1000
        self.monitor.record_request(latency_ms,
                                    is_error=response.status_code >= 500)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{latency_ms:.2f}"
        log.info("rid=%s %s %s -> %d (%.2f ms)", request_id, request.method,
                 request.url.path, response.status_code, latency_ms)
        return response
