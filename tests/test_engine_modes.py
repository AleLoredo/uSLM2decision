"""
Unit and Integration tests for OpenJEV Engine Modes (mock & real MLX)
"""

import os
import tempfile
import pytest
from core.config import OpenJEVConfig, ModelConfig
from core.engine_factory import create_engine
from core.engine_mock import MockInferenceEngine
from core.engine_mlx import MLXInferenceEngine
from core.model_scanner import init_canonical_heads

SAMPLE_QUERIES = [
    {
        "id": "q_bool",
        "type": "bool",
        "criteria": "Is the user requesting service cancellation?"
    },
    {
        "id": "q_choice",
        "type": "choice",
        "options": [
            {"id": "opt_billing", "description": "Billing inquiry"},
            {"id": "opt_tech", "description": "Technical support"}
        ]
    },
    {
        "id": "q_score",
        "type": "score",
        "min": 1,
        "max": 5,
        "rubric": "Urgency rating from 1 to 5"
    }
]

def test_mock_engine():
    cfg = OpenJEVConfig(mode="mock")
    engine = create_engine(cfg)
    assert isinstance(engine, MockInferenceEngine)
    assert not engine.is_ready()

    engine.load_model()
    assert engine.is_ready()

    decisions = engine.decide("Context text", SAMPLE_QUERIES)
    assert len(decisions) == 3
    assert decisions[0]["type"] == "bool"
    assert decisions[1]["type"] == "choice"
    assert decisions[2]["type"] == "score"

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

        engine.load_model()
        assert engine.is_ready()

        decisions = engine.decide("Account cancellation requested due to price", SAMPLE_QUERIES)
        assert len(decisions) == 3

        # Bool check
        d_bool = decisions[0]
        assert d_bool["type"] == "bool"
        assert isinstance(d_bool["value"], bool)
        assert 0.5 <= d_bool["confidence"] <= 1.0

        # Choice check
        d_choice = decisions[1]
        assert d_choice["type"] == "choice"
        assert d_choice["selected"] in ["opt_billing", "opt_tech"]
        assert pytest.approx(sum(d_choice["distribution"].values()), abs=1e-5) == 1.0

        # Score check
        d_score = decisions[2]
        assert d_score["type"] == "score"
        assert 1.0 <= d_score["score"] <= 5.0
        assert 1 <= d_score["discrete_level"] <= 5
        assert d_score["variance"] >= 0.0
