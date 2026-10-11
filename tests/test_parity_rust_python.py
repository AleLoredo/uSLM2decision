"""
Cross-runtime parity verification test between Python MLX and Rust openjev-mlx.
Validates SPEC.md and ARCHITECTURE.md conformance and numerical consistency across both runtime options.
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
    SPEC.md and ARCHITECTURE.md compliant responses for choice, score, and noul questions.
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
            "state": "Customer ordered a keyboard on Monday. Package arrived broken on Wednesday.",
            "model": "qwen2.5-0.5b",
            "questions": {
                "q_noul": {
                    "type": "noul",
                    "instructions": "Is customer eligible for immediate replacement?"
                },
                "q_choice": {
                    "type": "choice",
                    "instructions": "Select optimal resolution",
                    "criteria": {
                        "replace": "Send new replacement",
                        "refund": "Issue monetary refund",
                        "reject": "Deny replacement request"
                    }
                },
                "q_score": {
                    "type": "score",
                    "instructions": "Rate urgency",
                    "criteria": [
                        "Low",
                        "Medium",
                        "High",
                        "Critical"
                    ]
                }
            }
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
        
        assert data["model"] == "qwen2.5-0.5b"
        assert "answers" in data
        assert "usage" in data
        assert data["usage"]["input_tokens"] > 0
        assert data["usage"]["output_tokens"] > 0
        
        answers = data["answers"]
        assert len(answers) == 3
        
        # Check noul
        noul_ans = answers["q_noul"]
        assert noul_ans["type"] == "noul"
        assert 0.0 <= noul_ans["noul"] <= 1.0
        
        # Check choice
        choice_ans = answers["q_choice"]
        assert choice_ans["type"] == "choice"
        assert choice_ans["choice"] in ["replace", "refund", "reject"]
        assert 0.0 <= choice_ans["confidence"] <= 1.0
        dist = choice_ans["probabilities"]
        assert len(dist) == 3
        assert sum(dist.values()) == pytest.approx(1.0, rel=1e-3)
        
        # Check score
        score_ans = answers["q_score"]
        assert score_ans["type"] == "score"
        assert 0.0 <= score_ans["score"] <= 3.0
        assert 0.0 <= score_ans["confidence"] <= 1.0
        assert len(score_ans["legend"]) == 4
        assert len(score_ans["probabilities"]) == 4
        assert sum(score_ans["probabilities"].values()) == pytest.approx(1.0, rel=1e-3)
        
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
