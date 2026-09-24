"""
Phase 5 Unit & Integration Test Suite for Serving Engines (SPEC.md §4 & IMPLEMENTATION.md §8)
Verifies:
1. GET /healthz returns 200 OK.
2. GET /readyz returns 200 OK after warmup.
3. GET /metrics returns Prometheus metric format.
4. POST /v1/decide handles bool, choice, score queries with SPEC.md compliant responses.
5. Error handling for empty queries or invalid types returns 400 Bad Request.
"""

import pytest
from fastapi.testclient import TestClient
from server.edge_server import app, MODEL_VERSION, CALIBRATION_VERSION
from server.vllm_prefix_serving import VLLMPrefixServingEngine

client = TestClient(app)

def test_healthz_endpoint():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_readyz_endpoint():
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["model_version"] == MODEL_VERSION

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "opendecide_requests_total" in response.text

def test_decide_endpoint_valid_primitives():
    payload = {
        "context": "System security policy states 2FA is mandatory for admin users.",
        "queries": [
            {
                "id": "q1",
                "type": "bool",
                "criteria": "Is 2FA required for administrators?"
            },
            {
                "id": "q2",
                "type": "choice",
                "options": [
                    {"id": "opt_admin", "description": "Admin Users"},
                    {"id": "opt_guest", "description": "Guest Users"}
                ]
            },
            {
                "id": "q3",
                "type": "score",
                "min": 1,
                "max": 5,
                "rubric": "Security strictness 1-5"
            }
        ]
    }
    
    response = client.post("/v1/decide", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["model_version"] == MODEL_VERSION
    assert data["calibration_version"] == CALIBRATION_VERSION
    assert len(data["decisions"]) == 3
    
    # Check bool primitive
    d_bool = data["decisions"][0]
    assert d_bool["type"] == "bool"
    assert isinstance(d_bool["value"], bool)
    assert 0.5 <= d_bool["confidence"] <= 1.0
    
    # Check choice primitive
    d_choice = data["decisions"][1]
    assert d_choice["type"] == "choice"
    assert d_choice["selected"] in ["opt_admin", "opt_guest"]
    assert "distribution" in d_choice
    assert pytest.approx(sum(d_choice["distribution"].values()), abs=1e-5) == 1.0
    
    # Check score primitive
    d_score = data["decisions"][2]
    assert d_score["type"] == "score"
    assert 1.0 <= d_score["score"] <= 5.0
    assert 1 <= d_score["discrete_level"] <= 5
    assert d_score["variance"] >= 0.0

def test_decide_endpoint_invalid_payload():
    # Empty queries array -> 400 Bad Request
    res_empty = client.post("/v1/decide", json={"context": "test", "queries": []})
    assert res_empty.status_code == 400
    
    # Invalid query type -> 400 Bad Request
    res_invalid = client.post("/v1/decide", json={"context": "test", "queries": [{"type": "unknown"}]})
    assert res_invalid.status_code == 400

def test_vllm_prefix_serving_engine():
    engine = VLLMPrefixServingEngine()
    prefill_res = engine.prefill_state_context("Context document text")
    assert prefill_res["status"] == "prefilled"
    
    queries = [{"id": "q1", "type": "bool"}]
    exec_res = engine.execute_batch_queries(queries)
    assert len(exec_res) == 1
    assert exec_res[0]["prefix_cached"] is True
