"""
Unit tests for Standalone Rate Limiter Middleware (IMPLEMENTATION.md §8.1)
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from core.config import OpenJEVConfig, ServerConfig, RateLimitConfig
from server.middleware.rate_limiter import StandaloneRateLimiterMiddleware

def test_rate_limiter_allows_under_burst():
    app = FastAPI()
    cfg = OpenJEVConfig(
        server=ServerConfig(
            rate_limit=RateLimitConfig(enabled=True, max_burst=3, requests_per_second=1.0)
        )
    )
    app.add_middleware(StandaloneRateLimiterMiddleware, config=cfg)

    @app.post("/v1/decide")
    def decide():
        return {"status": "ok"}

    client = TestClient(app)

    # 3 bursts allowed
    for _ in range(3):
        res = client.post("/v1/decide")
        assert res.status_code == 200
        assert "X-RateLimit-Remaining" in res.headers

    # 4th exceeds burst capacity -> 429 Too Many Requests
    res4 = client.post("/v1/decide")
    assert res4.status_code == 429
    assert res4.headers.get("Retry-After") == "1"
    assert res4.json()["error"] == "TooManyRequests"

def test_rate_limiter_disabled_does_not_throttle():
    app = FastAPI()
    cfg = OpenJEVConfig(
        server=ServerConfig(
            rate_limit=RateLimitConfig(enabled=False, max_burst=1, requests_per_second=1.0)
        )
    )
    app.add_middleware(StandaloneRateLimiterMiddleware, config=cfg)

    @app.post("/v1/decide")
    def decide():
        return {"status": "ok"}

    client = TestClient(app)
    for _ in range(5):
        res = client.post("/v1/decide")
        assert res.status_code == 200
