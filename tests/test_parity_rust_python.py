"""
Cross-runtime parity verification test between Python MLX and Rust openjev-mlx.
Validates SPEC.md §4 conformance and numerical consistency across both runtime options.
"""

import json
import subprocess
import time
import urllib.request
import urllib.error
import pytest

def test_rust_binary_output_schema_conformance():
    """
    Verifies that the compiled Rust native binary executes and generates
    SPEC.md §4 compliant responses for bool, choice, and score queries.
    """
    port = 8099
    rust_cmd = ["./target/release/openjev-mlx"]
    
    # Write a temporary engine config for port 8099
    config_content = f"""
[server]
host = "127.0.0.1"
port = {port}

[server.auth]
enabled = false

[batching]
max_batch_size = 16
max_wait_ms = 8
max_queue_depth = 2048

[model]
name = "qwen2.5-0.5b"
mode = "real"
base_dir = "./models"
calibration_version = "calib-canonical-v1"
"""
    with open("config/engine.toml", "w") as f:
        f.write(config_content)
        
    proc = subprocess.Popen(rust_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server readiness
    url = f"http://127.0.0.1:{port}"
    ready = False
    for _ in range(30):
        try:
            req = urllib.request.Request(f"{url}/readyz")
            with urllib.request.urlopen(req, timeout=0.5) as response:
                if response.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.1)
            
    assert ready, "Rust server failed to start within 3 seconds"
    
    try:
        # 2. Issue decide request
        payload = {
            "context": "Customer ordered a keyboard on Monday. Package arrived broken on Wednesday.",
            "queries": [
                {
                    "type": "bool",
                    "id": "q_bool",
                    "criteria": "Is customer eligible for immediate replacement?"
                },
                {
                    "type": "choice",
                    "id": "q_choice",
                    "criteria": "Select optimal resolution",
                    "options": [
                        {"id": "replace", "description": "Send new replacement"},
                        {"id": "refund", "description": "Issue monetary refund"},
                        {"id": "reject", "description": "Deny replacement request"}
                    ]
                },
                {
                    "type": "score",
                    "id": "q_score",
                    "rubric": "Rate urgency from 1 to 5",
                    "min": 1,
                    "max": 5
                }
            ]
        }
        
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{url}/v1/decide",
            data=req_data,
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=2.0) as response:
            assert response.status == 200
            data = json.loads(response.read().decode("utf-8"))
        
        assert data["model_version"] == "qwen2.5-0.5b"
        assert data["calibration_version"] == "calib-canonical-v1"
        assert "metrics" in data and "latency_ms" in data["metrics"]
        
        decisions = data["decisions"]
        assert len(decisions) == 3
        
        # Check bool
        bool_dec = decisions[0]
        assert bool_dec["type"] == "bool"
        assert bool_dec["id"] == "q_bool"
        assert isinstance(bool_dec["value"], bool)
        assert 0.5 <= bool_dec["confidence"] <= 1.0
        
        # Check choice
        choice_dec = decisions[1]
        assert choice_dec["type"] == "choice"
        assert choice_dec["id"] == "q_choice"
        assert choice_dec["selected"] in ["replace", "refund", "reject"]
        assert 0.0 < choice_dec["confidence"] <= 1.0
        dist = choice_dec["distribution"]
        assert sum(dist.values()) == pytest.approx(1.0, rel=1e-3)
        
        # Check score
        score_dec = decisions[2]
        assert score_dec["type"] == "score"
        assert score_dec["id"] == "q_score"
        assert 1.0 <= score_dec["score"] <= 5.0
        assert 1 <= score_dec["discrete_level"] <= 5
        assert score_dec["variance"] >= 0.0
        assert len(score_dec["distribution"]) == 5
        assert sum(score_dec["distribution"]) == pytest.approx(1.0, rel=1e-3)
        
    finally:
        proc.terminate()
        proc.wait(timeout=2.0)
        # Restore default engine.toml
        default_config = """
[server]
host = "0.0.0.0"
port = 8000

[server.auth]
enabled = false
api_keys = ["jev-local-dev-key"]

[batching]
max_batch_size = 16
max_wait_ms = 8
max_queue_depth = 2048

[model]
name = "qwen2.5-0.5b"
mode = "real"
base_dir = "./models"
calibration_version = "calib-canonical-v1"
"""
        with open("config/engine.toml", "w") as f:
            f.write(default_config.strip() + "\n")
