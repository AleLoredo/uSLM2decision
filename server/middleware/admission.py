"""
OpenJEV Hardware Admission & Load Shedding Middleware
Enforces hard bounds on in-flight requests per IMPLEMENTATION.md §8.1 to protect Metal GPU and unified memory.
Returns 503 Service Unavailable with Retry-After header on queue saturation.
"""

import threading
from typing import Set
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from core.config import OpenJEVConfig

EXEMPT_PATHS: Set[str] = {
    "/healthz",
    "/readyz",
    "/metrics",
    "/docs",
    "/openapi.json"
}

class HardwareAdmissionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, config: OpenJEVConfig):
        super().__init__(app)
        self.config = config
        self._lock = threading.Lock()
        self._in_flight: int = 0

    @property
    def in_flight_count(self) -> int:
        with self._lock:
            return self._in_flight

    async def dispatch(self, request: Request, call_next):
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        admission_cfg = self.config.server.admission
        max_depth = admission_cfg.max_queue_depth
        retry_after = str(admission_cfg.retry_after_seconds)

        # 1. Admission Check & Load Shedding
        with self._lock:
            if self._in_flight >= max_depth:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "ServiceUnavailable",
                        "detail": f"Hardware admission rejected: pending queue at maximum capacity ({max_depth})."
                    },
                    headers={"Retry-After": retry_after}
                )
            self._in_flight += 1

        # 2. Execution and guaranteed cleanup
        try:
            response = await call_next(request)
            return response
        finally:
            with self._lock:
                self._in_flight = max(0, self._in_flight - 1)
