"""
Unit tests for Bearer Authentication Middleware (IMPLEMENTATION.md §8)
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from core.config import OpenJEVConfig, ServerConfig, AuthConfig
from server.middleware.auth import BearerAuthMiddleware

def create_test_app(auth_enabled: bool, keys: list):
    app = FastAPI()
    cfg = OpenJEVConfig(
        server=ServerConfig(
            auth=AuthConfig(enabled=auth_enabled, api_keys=keys)
        )
    )
    app.add_middleware(BearerAuthMiddleware, config=cfg)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.post("/v1/decide")
    def decide():
        return {"result": "authorized"}

    return app

def test_auth_disabled_allows_all():
    app = create_test_app(auth_enabled=False, keys=["secret-key"])
    client = TestClient(app)

    res = client.post("/v1/decide")
    assert res.status_code == 200
    assert res.json() == {"result": "authorized"}

def test_auth_enabled_rejects_missing_header():
    app = create_test_app(auth_enabled=True, keys=["secret-key"])
    client = TestClient(app)

    res = client.post("/v1/decide")
    assert res.status_code == 401
    assert "WWW-Authenticate" in res.headers
    assert res.json()["error"] == "Unauthorized"

def test_auth_enabled_rejects_invalid_token():
    app = create_test_app(auth_enabled=True, keys=["secret-key"])
    client = TestClient(app)

    res = client.post("/v1/decide", headers={"Authorization": "Bearer wrong-key"})
    assert res.status_code == 401
    assert res.json()["error"] == "Unauthorized"

def test_auth_enabled_accepts_valid_token():
    app = create_test_app(auth_enabled=True, keys=["secret-key"])
    client = TestClient(app)

    res = client.post("/v1/decide", headers={"Authorization": "Bearer secret-key"})
    assert res.status_code == 200
    assert res.json() == {"result": "authorized"}

def test_exempt_endpoints_bypass_auth():
    app = create_test_app(auth_enabled=True, keys=["secret-key"])
    client = TestClient(app)

    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}
