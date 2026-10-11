"""
OpenJEV Model Architecture Scanner & Canonical Heads Generator
Deterministically inspects Hugging Face config.json, validates heads.safetensors,
and initializes canonical heads when missing. Zero fuzzy matching.
"""

import os
import sys
import json
import argparse
from typing import Dict, Tuple, Optional
import numpy as np
from safetensors.numpy import save_file, load_file

CANONICAL_KEYS = [
    "bool_head.linear.weight",
    "bool_head.linear.bias",
    "choice_head.w_choice.weight",
    "score_head.w_score.weight",
    "score_head.w_score.bias"
]

def get_expected_shapes(hidden_dim: int) -> Dict[str, Tuple[int, ...]]:
    return {
        "bool_head.linear.weight": (1, hidden_dim),
        "bool_head.linear.bias": (1,),
        "choice_head.w_choice.weight": (hidden_dim, hidden_dim),
        "score_head.w_score.weight": (5, hidden_dim),
        "score_head.w_score.bias": (5,)
    }

def get_model_hidden_size(model_dir: str, default: int = 896) -> int:
    """Reads hidden_size or d_model from config.json in model_dir."""
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            hidden_size = data.get("hidden_size") or data.get("d_model")
            if hidden_size and isinstance(hidden_size, int):
                return hidden_size
        except Exception as e:
            print(f"[OpenJEV Scanner] Warning: Could not parse {config_path}: {e}")
    return default

def verify_canonical_heads(heads_path: str, expected_hidden_dim: int) -> None:
    """
    Verifies that heads.safetensors contains all canonical keys with exact shapes.
    Raises ValueError with descriptive diagnostics on failure.
    """
    if not os.path.isfile(heads_path):
        raise FileNotFoundError(f"Heads file not found: {heads_path}")

    tensors = load_file(heads_path)
    expected_shapes = get_expected_shapes(expected_hidden_dim)

    # 1. Key existence check
    missing_keys = [k for k in CANONICAL_KEYS if k not in tensors]
    if missing_keys:
        raise ValueError(
            f"Strict Schema Mismatch: Missing required keys in '{heads_path}': {missing_keys}. "
            f"Expected canonical keys: {CANONICAL_KEYS}"
        )

    # 2. Shape assertion check
    for k, expected_shape in expected_shapes.items():
        actual_shape = tuple(tensors[k].shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Dimension Mismatch: Key '{k}' has shape {actual_shape} in '{heads_path}', "
                f"but backbone hidden_size={expected_hidden_dim} requires {expected_shape}."
            )

def init_canonical_heads(model_dir: str, hidden_dim: Optional[int] = None, force: bool = False) -> str:
    """
    Creates canonical heads.safetensors and heads_metadata.json in model_dir.
    If already existing and valid, returns the existing path unless force=True.
    """
    os.makedirs(model_dir, exist_ok=True)
    heads_path = os.path.join(model_dir, "heads.safetensors")

    if hidden_dim is None:
        hidden_dim = get_model_hidden_size(model_dir, default=896)

    # If already exists and valid, skip generation unless force is True
    if os.path.isfile(heads_path) and not force:
        try:
            verify_canonical_heads(heads_path, hidden_dim)
            return heads_path
        except ValueError as e:
            print(f"[OpenJEV Scanner] Warning: Existing heads file invalid ({e}). Regenerating...")

    expected_shapes = get_expected_shapes(hidden_dim)
    rng = np.random.default_rng(42)

    # Initialize calibrated canonical weights
    # Bool head: small Gaussian weights + zero bias
    w_bool = (rng.normal(0, 0.02, size=expected_shapes["bool_head.linear.weight"])).astype(np.float32)
    b_bool = np.zeros(expected_shapes["bool_head.linear.bias"], dtype=np.float32)

    # Choice head: identity + small noise projection
    w_choice = (np.eye(hidden_dim, dtype=np.float32) + rng.normal(0, 0.01, size=(hidden_dim, hidden_dim))).astype(np.float32)

    # Score head: small Gaussian weights + zero bias
    w_score = (rng.normal(0, 0.02, size=expected_shapes["score_head.w_score.weight"])).astype(np.float32)
    b_score = np.zeros(expected_shapes["score_head.w_score.bias"], dtype=np.float32)

    tensors = {
        "bool_head.linear.weight": w_bool,
        "bool_head.linear.bias": b_bool,
        "choice_head.w_choice.weight": w_choice,
        "score_head.w_score.weight": w_score,
        "score_head.w_score.bias": b_score
    }

    save_file(tensors, heads_path)

    # Write metadata manifest
    metadata = {
        "format_version": "1.0",
        "openjev_version": "2.0.0",
        "hidden_dim": hidden_dim,
        "status": "canonical_initialized",
        "tensors": {k: list(v) for k, v in expected_shapes.items()}
    }
    meta_path = os.path.join(model_dir, "heads_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OpenJEV Scanner] Successfully initialized canonical heads: {heads_path} (hidden_dim={hidden_dim})")
    return heads_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenJEV Model Scanner & Canonical Heads Generator")
    parser.add_argument("--model-dir", required=True, help="Path to model directory")
    parser.add_argument("--init-heads", action="store_true", help="Initialize canonical heads if missing or invalid")
    parser.add_argument("--verify", action="store_true", help="Verify canonical heads against config.json")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-generation of canonical heads even if present")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Explicit hidden dimension (optional)")

    args = parser.parse_args()
    h_dim = args.hidden_dim or get_model_hidden_size(args.model_dir)

    if args.init_heads:
        init_canonical_heads(args.model_dir, hidden_dim=h_dim, force=args.force)
    elif args.verify:
        h_path = os.path.join(args.model_dir, "heads.safetensors")
        verify_canonical_heads(h_path, h_dim)
        print(f"[OpenJEV Scanner] Verification PASSED for {h_path} (hidden_dim={h_dim})")
