"""
Unit tests for core/config.py
"""

import os
import tempfile
import pytest
from core.config import OpenJEVConfig, ModelConfig, load_config, save_config

def test_default_config_loading():
    config = load_config("non_existent_file.yaml")
    assert config.mode == "mock"
    assert config.model.active_model == "qwen2.5-0.5b"
    assert config.server.port == 8000

def test_config_save_and_reload():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_config_file = os.path.join(tmpdir, "config.yaml")
        cfg = OpenJEVConfig(
            mode="real",
            model=ModelConfig(
                active_model="custom-0.5b",
                backbone_path="/tmp/backbone",
                heads_path="/tmp/heads.safetensors"
            )
        )
        save_config(cfg, tmp_config_file)
        
        reloaded = load_config(tmp_config_file)
        assert reloaded.mode == "real"
        assert reloaded.model.active_model == "custom-0.5b"
        assert reloaded.model.backbone_path == "/tmp/backbone"

def test_validate_real_paths():
    cfg = OpenJEVConfig(
        mode="real",
        model=ModelConfig(
            backbone_path="/non/existent/path/backbone",
            heads_path="/non/existent/path/heads.safetensors"
        )
    )
    missing = cfg.validate_real_paths()
    assert len(missing) == 2
    assert "backbone_path not found" in missing[0]
    assert "heads_path not found" in missing[1]

def test_mock_mode_skips_path_validation():
    cfg = OpenJEVConfig(
        mode="mock",
        model=ModelConfig(
            backbone_path="/non/existent/path/backbone",
            heads_path="/non/existent/path/heads.safetensors"
        )
    )
    missing = cfg.validate_real_paths()
    assert len(missing) == 0
