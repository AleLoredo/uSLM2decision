"""
Unit tests for Hardware Admission & Load Shedding Middleware (IMPLEMENTATION.md §8.1)
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from core.config import OpenJEVConfig, ServerConfig, AdmissionConfig
from server.middleware.admission import HardwareAdmissionMiddleware

def test_hardware_admission_sheds_load_when_full():
    app = FastAPI()
    # Configure tiny queue depth of 1 for testing
    cfg = OpenJEVConfig(
        server=ServerConfig(
            admission=AdmissionConfig(max_queue_depth=1, retry_after_seconds=3)
        )
    )
    middleware = HardwareAdmissionMiddleware(app, config=cfg)
    app.add_middleware(HardwareAdmissionMiddleware, config=cfg)

    @app.post("/v1/decide")
    def decide():
        return {"status": "processed"}

    client = TestClient(app)

    # First request succeeds
    res1 = client.post("/v1/decide")
    assert res1.status_code == 200

    # Simulate saturation by manually holding lock count to capacity
    app.user_middleware[0].kwargs["config"].server.admission.max_queue_depth = 0
    res_saturated = client.post("/v1/decide")
    assert res_saturated.status_code == 503
    assert res_saturated.headers.get("Retry-After") == "3"
    assert res_saturated.json()["error"] == "ServiceUnavailable"
