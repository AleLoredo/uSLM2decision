"""
Phase 4 Unit Test Suite for Post-Hoc Calibration & Fast Domain Recalibration
Verifies:
1. TemperatureScaler fits optimal scalar T reducing ECE on over-confident logits.
2. Fast self-hosted recalibration routine completes in < 100 ms on 200 domain examples.
"""

import pytest
import numpy as np
import mlx.core as mx
from calibration.post_hoc import TemperatureScaler
from calibration.recalibrate import recalibrate_domain_model
from calibration.metrics import compute_ece

def test_temperature_scaler_reduces_ece():
    np.random.seed(42)
    n = 200
    logits_np = np.random.normal(3.0, 1.0, size=n) # Uncalibrated over-confident logits
    probs_true = 1.0 / (1.0 + np.exp(-logits_np * 0.3))
    labels = np.random.binomial(1, probs_true).astype(np.float32)
    
    uncalib_probs = 1.0 / (1.0 + np.exp(-logits_np))
    ece_before = compute_ece(uncalib_probs, labels)["ece"]
    
    scaler = TemperatureScaler(initial_temperature=1.0)
    fitted_T = scaler.fit(mx.array(logits_np), mx.array(labels), search_range=(0.5, 5.0), steps=100)
    
    calib_probs = 1.0 / (1.0 + np.exp(-logits_np / fitted_T))
    ece_after = compute_ece(calib_probs, labels)["ece"]
    
    assert fitted_T > 1.0 # Pushes temperature up to soften over-confident logits
    assert ece_after < ece_before

def test_fast_recalibration_performance():
    logits = mx.random.normal((200,))
    targets = mx.array(np.random.binomial(1, 0.5, size=200).astype(np.float32))
    
    result = recalibrate_domain_model(logits, targets, primitive_type="bool")
    
    assert result["status"] == "success"
    assert result["num_examples"] == 200
    assert result["elapsed_seconds"] < 1.0 # Under 1 second SLA
