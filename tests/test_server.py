"""
Phase 5 Unit & Integration Test Suite for Serving Engines (SPEC.md §4 & ARCHITECTURE.md)
Verifies:
1. GET /healthz returns 200 OK.
2. GET /readyz returns 200 OK after warmup.
3. GET /metrics returns Prometheus metric format.
4. POST /v1/decide handles choice, score, noul questions with SPEC.md canonical response.
5. Structured JSON states are correctly accepted.
6. Legacy schemas ('context', 'queries', 'bool') are rejected with 422 Unprocessable Entity.
7. Error handling for empty questions returns 400 Bad Request.
"""

import pytest
from fastapi.testclient import TestClient
from server.edge_server import app, MODEL_VERSION
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
    assert "openjev_requests_total" in response.text

def test_decide_endpoint_canonical_primitives():
    payload = {
        "state": "Hi, I've been trying to connect my Stripe account for 3 days and the integration keeps failing. I'm losing sales. Please help ASAP.",
        "model": "jev-latest",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this",
                "criteria": {
                    "billing": "Payment or subscription issues",
                    "technical": "Bugs or integration problems",
                    "sales": "Pricing or account questions"
                }
            },
            "frustration": {
                "type": "score",
                "instructions": "How frustrated the customer appears",
                "criteria": [
                    "Calm, just stating facts",
                    "Frustrated but civil",
                    "Very angry, strong language"
                ]
            },
            "is_urgent": {
                "type": "noul",
                "instructions": "The message conveys urgency or time-sensitivity"
            }
        }
    }

    response = client.post("/v1/decide", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["model"].startswith("jev-")
    assert "answers" in data
    assert "usage" in data
    assert data["usage"]["input_tokens"] > 0
    assert data["usage"]["output_tokens"] > 0

    answers = data["answers"]
    assert len(answers) == 3

    # Check choice primitive
    ans_choice = answers["department"]
    assert ans_choice["type"] == "choice"
    assert ans_choice["choice"] in ["billing", "technical", "sales"]
    assert 0.0 <= ans_choice["confidence"] <= 1.0
    assert "probabilities" in ans_choice
    assert len(ans_choice["probabilities"]) == 3
    assert pytest.approx(sum(ans_choice["probabilities"].values()), abs=1e-4) == 1.0

    # Check score primitive
    ans_score = answers["frustration"]
    assert ans_score["type"] == "score"
    assert 0.0 <= ans_score["score"] <= 2.0
    assert 0.0 <= ans_score["confidence"] <= 1.0
    assert "legend" in ans_score
    assert len(ans_score["legend"]) == 3
    assert "probabilities" in ans_score
    assert pytest.approx(sum(ans_score["probabilities"].values()), abs=1e-4) == 1.0

    # Check noul primitive
    ans_noul = answers["is_urgent"]
    assert ans_noul["type"] == "noul"
    assert ans_noul["noul"] in [0, 1]
    assert isinstance(ans_noul["noul"], int)

def test_decide_endpoint_structured_json_state():
    payload = {
        "state": {
            "customer_id": "cust_1234",
            "tier": "enterprise",
            "recent_tickets": 4,
            "message": "Service outage affecting checkout"
        },
        "questions": {
            "critical_outage": {
                "type": "noul",
                "instructions": "Does this indicate a revenue-impacting outage?"
            }
        }
    }
    response = client.post("/v1/decide", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answers" in data
    assert "critical_outage" in data["answers"]
    assert data["answers"]["critical_outage"]["type"] == "noul"
    assert data["answers"]["critical_outage"]["noul"] in [0, 1]

def test_decide_endpoint_legacy_payloads_rejected_422():
    # Legacy 'context' and 'queries' schema must be rejected (Zero backward compatibility)
    legacy_payload = {
        "context": "Customer requests refund",
        "queries": {
            "valid": {"type": "bool", "criteria": "Is refund valid?"}
        }
    }
    res = client.post("/v1/decide", json=legacy_payload)
    assert res.status_code == 422

    # Legacy 'bool' type in questions must be rejected
    bool_payload = {
        "state": "Customer requests refund",
        "questions": {
            "valid": {"type": "bool", "criteria": "Is refund valid?"}
        }
    }
    res_bool = client.post("/v1/decide", json=bool_payload)
    assert res_bool.status_code == 422

def test_decide_endpoint_empty_questions_400():
    # Empty questions map -> 400 Bad Request
    res_empty = client.post("/v1/decide", json={"state": "test", "questions": {}})
    assert res_empty.status_code == 400
    assert "Questions map cannot be empty" in res_empty.json()["detail"]

def test_vllm_prefix_serving_engine():
    engine = VLLMPrefixServingEngine()
    prefill_res = engine.prefill_state_context("Context document text")
    assert prefill_res["status"] == "prefilled"

    queries = [{"id": "q1", "type": "noul"}]
    exec_res = engine.execute_batch_queries(queries)
    assert len(exec_res) == 1
    assert exec_res[0]["prefix_cached"] is True
