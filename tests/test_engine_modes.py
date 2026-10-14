"""
Unit and Integration tests for OpenJEV Engine Modes (mock & real MLX)
Conforming to System 1 architecture (SPEC.md and ARCHITECTURE.md)
"""

import os
import tempfile
import pytest
from core.config import OpenJEVConfig, ModelConfig
from core.engine_factory import create_engine
from core.engine_mock import MockInferenceEngine
from core.engine_mlx import MLXInferenceEngine
from core.model_scanner import init_canonical_heads

SAMPLE_QUESTIONS = {
    "is_cancellation": {
        "type": "noul",
        "instructions": "Is the user requesting service cancellation?"
    },
    "department": {
        "type": "choice",
        "instructions": "Select handling department",
        "criteria": {
            "opt_billing": "Billing inquiry",
            "opt_tech": "Technical support"
        }
    },
    "urgency": {
        "type": "score",
        "instructions": "Urgency rating from 0 to 4",
        "criteria": [
            "Level 0 - lowest",
            "Level 1 - low",
            "Level 2 - medium",
            "Level 3 - high",
            "Level 4 - critical"
        ]
    }
}

def test_mock_engine():
    cfg = OpenJEVConfig(mode="mock")
    engine = create_engine(cfg)
    assert isinstance(engine, MockInferenceEngine)
    assert not engine.is_ready()

    engine.load_model()
    assert engine.is_ready()

    res = engine.decide("Context text", SAMPLE_QUESTIONS)
    assert "answers" in res
    assert "usage" in res
    answers = res["answers"]
    assert len(answers) == 3
    assert answers["is_cancellation"]["type"] == "noul"
    assert answers["department"]["type"] == "choice"
    assert answers["urgency"]["type"] == "score"

@pytest.mark.requires_mlx
def test_mlx_engine_real_mode():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize canonical heads in tmpdir
        heads_path = init_canonical_heads(tmpdir, hidden_dim=896)

        cfg = OpenJEVConfig(
            mode="real",
            model=ModelConfig(
                active_model="test-mlx-model",
                backbone_path=tmpdir,
                heads_path=heads_path,
                backend="mlx"
            )
        )

        engine = create_engine(cfg)
        assert isinstance(engine, MLXInferenceEngine)

        engine.load_model(require_backbone=False)
        assert engine.is_ready()

        res = engine.decide("Account cancellation requested due to price", SAMPLE_QUESTIONS)
        assert "answers" in res
        assert "usage" in res
        answers = res["answers"]
        assert len(answers) == 3

        # Noul check
        ans_noul = answers["is_cancellation"]
        assert ans_noul["type"] == "noul"
        assert ans_noul["noul"] in [0, 1]

        # Choice check
        ans_choice = answers["department"]
        assert ans_choice["type"] == "choice"
        assert ans_choice["choice"] in ["opt_billing", "opt_tech"]
        assert pytest.approx(sum(ans_choice["probabilities"].values()), abs=1e-4) == 1.0

        # Score check
        ans_score = answers["urgency"]
        assert ans_score["type"] == "score"
        assert 0.0 <= ans_score["score"] <= 4.0
        assert len(ans_score["legend"]) == 5
        assert pytest.approx(sum(ans_score["probabilities"].values()), abs=1e-4) == 1.0
