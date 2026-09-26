"""
OpenJEV Trace & Request-ID Middleware
Injects correlation IDs per request per IMPLEMENTATION.md §8 and records structured latency logs.
"""

import time
import uuid
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("openjev.trace")

class RequestTraceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        
        # 1. Retrieve or generate correlation ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
            
        request.state.request_id = request_id

        # 2. Process request
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 3. Inject correlation ID into response headers
        response.headers["X-Request-ID"] = request_id

        # 4. Structured log for non-health endpoints
        if not request.url.path.startswith(("/healthz", "/readyz", "/metrics")):
            logger.info(
                f"[TRACE] id={request_id} path={request.url.path} status={response.status_code} latency_ms={elapsed_ms:.2f}"
            )

        return response
