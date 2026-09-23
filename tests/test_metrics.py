import numpy as np
import pytest
from eval_harness.metrics import (
    compute_ece,
    compute_brier_decomposition,
    compute_aurc,
    compute_ranked_probability_score
)

def test_perfect_calibration_ece():
    # Model that predicts 0.8 confidence and is correct 80% of the time,
    # and 0.2 confidence and is correct 20% of the time.
    confs = np.array([0.8] * 80 + [0.8] * 20 + [0.2] * 20 + [0.2] * 80)
    labels = np.array([1] * 80 + [0] * 20 + [1] * 20 + [0] * 80)
    
    res = compute_ece(confs, labels, n_bins=10, scheme="equal_width")
    assert res["ece"] < 0.05
    assert res["scheme"] == "equal_width"

def test_adaptive_binning_ece():
    confs = np.random.uniform(0.5, 1.0, size=200)
    labels = np.random.binomial(1, 0.7, size=200)
    
    res_eq = compute_ece(confs, labels, n_bins=10, scheme="equal_width")
    res_ad = compute_ece(confs, labels, n_bins=10, scheme="adaptive")
    
    assert "ece" in res_eq and "ece" in res_ad
    assert len(res_ad["bin_counts"]) == 10

def test_brier_score_decomposition():
    probs = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.1, 0.9],
        [0.2, 0.8]
    ])
    labels = np.array([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1]
    ])
    
    decomp = compute_brier_decomposition(probs, labels)
    assert decomp["brier_score"] < 0.1
    assert "reliability" in decomp
    assert "resolution" in decomp
    assert "uncertainty" in decomp

def test_aurc_abstention():
    # Highly confident correct predictions followed by uncertain wrong predictions
    confs = np.array([0.99, 0.95, 0.90, 0.85, 0.60, 0.55])
    correctness = np.array([True, True, True, True, False, False])
    
    res = compute_aurc(confs, correctness)
    assert 0.0 <= res["aurc"] <= 1.0
    # Perfect selective prediction should have low AURC
    assert res["aurc"] < 0.25

def test_ranked_probability_score_ordinal():
    # 5 ordinal levels: 0, 1, 2, 3, 4
    # Case 1: Sharp prediction on level 2
    sharp_probs = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]])
    true_level = np.array([2])
    res_sharp = compute_ranked_probability_score(sharp_probs, true_level)
    assert res_sharp["mean_rps"] == 0.0 # Perfect match
    
    # Case 2: Near miss vs distant miss
    near_miss_probs = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]]) # Predicted 1, true 2
    far_miss_probs = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])  # Predicted 0, true 2
    
    res_near = compute_ranked_probability_score(near_miss_probs, true_level)
    res_far = compute_ranked_probability_score(far_miss_probs, true_level)
    
    # RPS must penalize distant miss more than near miss (ordinal property!)
    assert res_near["mean_rps"] < res_far["mean_rps"]
