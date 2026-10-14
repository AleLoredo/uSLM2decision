"""
Multi-Model Non-Regression & Compatibility Verification Suite
Verifies that all three supported model families:
1. Qwen2.5-0.5B (d=896, Llama-like homogeneous)
2. Gemma-2-2B (d=2304, Google-legacy)
3. OpenELM-450M (d=1536, Apple layer-varying non-uniform)

can be sequentially loaded and executed on Metal GPU with zero cross-contamination,
exact canonical head dimensions, and strict SPEC.md compliance.
"""

import os
import pytest
from core.config import OpenJEVConfig, ModelConfig
from core.engine_mlx import MLXInferenceEngine
from core.model_scanner import verify_canonical_heads

MODELS = [
    {
        "name": "qwen2.5-0.5b",
        "dir": "./models/qwen2.5-0.5b",
        "expected_dim": 896,
        "expected_layers": 24
    },
    {
        "name": "gemma-2-2b",
        "dir": "./models/gemma-2-2b",
        "expected_dim": 2304,
        "expected_layers": 26
    },
    {
        "name": "openelm-450m",
        "dir": "./models/openelm-450m",
        "expected_dim": 1536,
        "expected_layers": 20
    }
]

@pytest.mark.parametrize("model_info", MODELS, ids=lambda m: m["name"])
def test_model_heads_dimension_integrity(model_info):
    """Verifies that each model's heads.safetensors matches its exact expected hidden_dim."""
    heads_path = os.path.join(model_info["dir"], "heads.safetensors")
    assert os.path.isfile(heads_path), f"heads.safetensors missing for {model_info['name']}"
    verify_canonical_heads(heads_path, expected_hidden_dim=model_info["expected_dim"])

@pytest.mark.parametrize("model_info", MODELS, ids=lambda m: m["name"])
def test_model_inference_engine_lifecycle(model_info):
    """Tests loading, dimension binding, and decision execution for each model."""
    cfg = OpenJEVConfig(
        mode="real",
        model=ModelConfig(
            active_model=model_info["name"],
            backbone_path=model_info["dir"],
            heads_path=os.path.join(model_info["dir"], "heads.safetensors"),
            calibration_path=os.path.join(model_info["dir"], "calibration.json"),
            backend="mlx"
        )
    )

    engine = MLXInferenceEngine(cfg)
    engine.load_model()

    assert engine.is_ready()
    assert engine.hidden_dim == model_info["expected_dim"]
    assert len(engine.backbone.layers) == model_info["expected_layers"]

    # Execute canonical decision payload
    state = "The user has reported a recurrent crash on application startup after update."
    questions = {
        "is_urgent": {
            "type": "noul",
            "instructions": "Determine if the request requires immediate action"
        },
        "category": {
            "type": "choice",
            "instructions": "Route to the appropriate support desk",
            "criteria": {
                "support": "General customer service and troubleshooting",
                "billing": "Invoices and subscription payment questions",
                "engineering": "Platform bugs and crash diagnostics"
            }
        },
        "severity": {
            "type": "score",
            "instructions": "Rate severity from minor to critical",
            "criteria": [
                "Minor inconvenience",
                "Workaround available",
                "Critical crash preventing core workflow"
            ]
        }
    }

    result = engine.decide(state, questions)

    assert "answers" in result
    answers = result["answers"]

    # Verify noul
    assert "is_urgent" in answers
    assert answers["is_urgent"]["type"] == "noul"
    assert answers["is_urgent"]["noul"] in (0, 1)

    # Verify choice
    assert "category" in answers
    assert answers["category"]["type"] == "choice"
    assert answers["category"]["choice"] in ["support", "billing", "engineering"]
    assert "probabilities" in answers["category"]
    assert len(answers["category"]["probabilities"]) == 3

    # Verify score
    assert "severity" in answers
    assert answers["severity"]["type"] == "score"
    assert 0.0 <= answers["severity"]["score"] <= 2.0
    assert "probabilities" in answers["severity"]
    assert len(answers["severity"]["probabilities"]) == 3
