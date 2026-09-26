"""
OpenJEV Standalone Rate Limiter Middleware (Token Bucket)
Implements optional in-process rate limiting per IMPLEMENTATION.md §8.1 for edge deployments without a reverse proxy.
Returns 429 Too Many Requests when token bucket is exhausted.
"""

import time
import threading
from typing import Dict, Tuple, Set
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

class TokenBucket:
    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_update = time.monotonic()

    def consume(self, amount: float = 1.0) -> Tuple[bool, float]:
        now = time.monotonic()
        elapsed = now - self.last_update
        self.last_update = now

        # Replenish tokens based on elapsed time
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)

        if self.tokens >= amount:
            self.tokens -= amount
            return True, self.tokens
        return False, self.tokens

class StandaloneRateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, config: OpenJEVConfig):
        super().__init__(app)
        self.config = config
        self._lock = threading.Lock()
        self._buckets: Dict[str, TokenBucket] = {}

    def _get_client_id(self, request: Request) -> str:
        # Client IP or forwarded-for
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown_client"

    async def dispatch(self, request: Request, call_next):
        # Skip if rate limiting disabled or path is exempt
        if not self.config.server.rate_limit.enabled or request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        rl_cfg = self.config.server.rate_limit
        client_id = self._get_client_id(request)

        with self._lock:
            if client_id not in self._buckets:
                self._buckets[client_id] = TokenBucket(
                    capacity=float(rl_cfg.max_burst),
                    refill_rate=float(rl_cfg.requests_per_second)
                )
            bucket = self._buckets[client_id]
            allowed, remaining = bucket.consume(1.0)

        limit_headers = {
            "X-RateLimit-Limit": str(int(rl_cfg.max_burst)),
            "X-RateLimit-Remaining": str(max(0, int(remaining)))
        }

        if not allowed:
            retry_after = "1"
            return JSONResponse(
                status_code=429,
                content={
                    "error": "TooManyRequests",
                    "detail": "Rate limit exceeded. Too many requests submitted in current window."
                },
                headers={
                    "Retry-After": retry_after,
                    **limit_headers
                }
            )

        response = await call_next(request)
        for k, v in limit_headers.items():
            response.headers[k] = v
        return response
