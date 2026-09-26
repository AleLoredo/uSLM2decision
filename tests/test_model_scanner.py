"""
Unit tests for core/model_scanner.py
"""

import os
import json
import tempfile
import pytest
from core.model_scanner import (
    get_model_hidden_size,
    init_canonical_heads,
    verify_canonical_heads,
    CANONICAL_KEYS
)

def test_get_model_hidden_size_from_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = os.path.join(tmpdir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump({"hidden_size": 1024, "model_type": "test_model"}, f)
        
        dim = get_model_hidden_size(tmpdir)
        assert dim == 1024

def test_init_and_verify_canonical_heads():
    with tempfile.TemporaryDirectory() as tmpdir:
        heads_path = init_canonical_heads(tmpdir, hidden_dim=896)
        assert os.path.isfile(heads_path)
        assert os.path.isfile(os.path.join(tmpdir, "heads_metadata.json"))

        # Verification passes
        verify_canonical_heads(heads_path, expected_hidden_dim=896)

def test_verify_canonical_heads_dimension_mismatch():
    with tempfile.TemporaryDirectory() as tmpdir:
        heads_path = init_canonical_heads(tmpdir, hidden_dim=896)

        # Mismatch: asking for 1536 when file was generated with 896
        with pytest.raises(ValueError, match="Dimension Mismatch"):
            verify_canonical_heads(heads_path, expected_hidden_dim=1536)
