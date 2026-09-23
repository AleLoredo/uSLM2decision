"""
Phase 1 Unit & Regression Test Suite for Decision Heads (AGENTS.md DoD)
Verifies:
1. Criteria Sensitivity Test: Choice queries with identical options but different criteria produce DIFFERENT distributions.
2. Score Ordinality Test: Ordinal loss penalizes off-by-2 error strictly more than off-by-1 error.
3. Bool Head Calibration Bounds: Confidence is always in [0.5, 1.0].
"""

import pytest
import mlx.core as mx
from core.heads import BoolHead, ChoiceHead, ScoreHead
from core.losses import ordinal_score_loss

def test_bool_head_bounds_and_tie_break():
    head = BoolHead(hidden_dim=32)
    h_q = mx.random.normal((4, 32))
    
    values, confidences, p_true = head(h_q)
    mx.eval(values, confidences, p_true)
    
    assert values.shape == (4,)
    assert confidences.shape == (4,)
    assert bool(mx.all(confidences >= 0.5)) and bool(mx.all(confidences <= 1.0))
    
    # Check exact 0.5 tie breaking (logit = 0.0)
    head.linear.bias = mx.zeros_like(head.linear.bias)
    head.linear.weight = mx.zeros_like(head.linear.weight)
    h_zero = mx.zeros((1, 32))
    v_zero, c_zero, p_zero = head(h_zero)
    mx.eval(v_zero, c_zero, p_zero)
    assert float(p_zero[0]) == 0.5
    assert bool(v_zero[0]) == False # Tie breaking rule
    assert float(c_zero[0]) == 0.5

def test_choice_head_criteria_sensitivity_regression():
    """
    CRITICAL REGRESSION TEST (AGENTS.md Phase 1 DoD & ARCHITECTURE.md §3.2)
    Two choice queries with IDENTICAL option embeddings but DIFFERENT query criteria
    (different h_q1, h_q2) MUST produce DIFFERENT option probability distributions!
    """
    mx.random.seed(42)
    hidden_dim = 64
    head = ChoiceHead(hidden_dim=hidden_dim)
    
    # Same 3 options
    e_options = mx.random.normal((3, hidden_dim))
    
    # Query 1 vs Query 2 (different criteria representations)
    h_q1 = mx.random.normal((1, hidden_dim))
    h_q2 = mx.random.normal((1, hidden_dim))
    
    _, _, probs1 = head(h_q1, e_options)
    _, _, probs2 = head(h_q2, e_options)
    mx.eval(probs1, probs2)
    
    probs1_list = [float(p) for p in probs1[0]]
    probs2_list = [float(p) for p in probs2[0]]
    
    assert probs1_list != probs2_list, (
        "REGRESSION BUG DETECTED: Choice head produced identical distributions "
        "for different query criteria!"
    )
    assert pytest.approx(sum(probs1_list), abs=1e-5) == 1.0
    assert pytest.approx(sum(probs2_list), abs=1e-5) == 1.0

def test_ordinal_score_head_and_loss_property():
    """
    Verifies score head calculation and ordinal loss property:
    An off-by-2 prediction MUST produce strictly higher loss than an off-by-1 prediction.
    """
    hidden_dim = 64
    num_levels = 5
    head = ScoreHead(hidden_dim=hidden_dim, num_levels=num_levels)
    
    h_q = mx.random.normal((2, hidden_dim))
    out = head(h_q, min_val=1, max_val=5)
    mx.eval(out["score"], out["discrete_level"], out["variance"], out["distribution"])
    
    assert out["score"].shape == (2,)
    assert out["discrete_level"].shape == (2,)
    assert out["variance"].shape == (2,)
    assert out["distribution"].shape == (2, 5)
    assert bool(mx.all(out["variance"] >= 0.0))
    
    # Ordinal Loss Property Test
    target = mx.array([2]) # Ground truth index 2 (Level 3)
    
    # Sharp mass on index 2 (Level 3 - Exact match)
    p_exact = mx.array([[0.0, 0.0, 1.0, 0.0, 0.0]])
    # Mass on index 1 (Level 2 - Off by 1)
    p_off_by_1 = mx.array([[0.0, 1.0, 0.0, 0.0, 0.0]])
    # Mass on index 0 (Level 1 - Off by 2)
    p_off_by_0 = mx.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
    
    loss_exact = float(ordinal_score_loss(p_exact, target, num_levels=5))
    loss_off1 = float(ordinal_score_loss(p_off_by_1, target, num_levels=5))
    loss_off2 = float(ordinal_score_loss(p_off_by_0, target, num_levels=5))
    
    assert loss_exact < loss_off1, "Exact match loss must be smaller than off-by-1 loss"
    assert loss_off1 < loss_off2, "Off-by-1 loss must be strictly smaller than off-by-2 loss (Ordinality violation!)"
