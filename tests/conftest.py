"""
Pytest configuration and global hooks for OpenJEV.
Ensures test environment defaults to mock mode unless explicitly configured,
and provides markers for conditional skipping of tests requiring MLX or local model weights.
"""

import os
import pytest

# Ensure tests import edge_server and configuration in mock mode by default
os.environ.setdefault("OPENJEV_MODE", "mock")

def _has_mlx() -> bool:
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False

def _has_model_heads(model_name: str) -> bool:
    heads_file = os.path.join("models", model_name, "heads.safetensors")
    return os.path.isfile(heads_file)

def _has_model_weights(model_name: str) -> bool:
    model_dir = os.path.join("models", model_name)
    if not os.path.isdir(model_dir):
        return False
    config_file = os.path.join(model_dir, "config.json")
    heads_file = os.path.join(model_dir, "heads.safetensors")
    if not (os.path.isfile(config_file) and os.path.isfile(heads_file)):
        return False
    # Check for actual transformer backbone weight files
    has_backbone = any(
        (f.endswith(".safetensors") and f != "heads.safetensors")
        or f.endswith(".bin")
        or f == "model.safetensors.index.json"
        for f in os.listdir(model_dir)
    )
    return has_backbone

def pytest_runtest_setup(item):
    for mark in item.iter_markers(name="requires_mlx"):
        if not _has_mlx():
            pytest.skip("Test requires Apple MLX framework and Metal GPU")

    for mark in item.iter_markers(name="requires_weights"):
        for model_name in mark.args:
            if not _has_model_weights(model_name):
                pytest.skip(
                    f"Test requires local weights for '{model_name}' (missing in ./models/{model_name}/). "
                    f"Run ./scripts/setup_model.sh to download."
                )
