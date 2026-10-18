"""
Phase 3 Unit Test Suite for Calibration Loss Shaping & Focal-Brier Integration
Verifies:
1. Focal binary loss calculation and non-negativity.
2. Brier score loss calculation.
3. Combined focal + Brier loss formulation.
"""

import pytest
import mlx.core as mx
from calibration.loss_shaping import (
    focal_binary_loss,
    brier_score_loss,
    combined_focal_brier_loss
)

def test_focal_binary_loss_properties():
    p_true = mx.array([0.9, 0.1, 0.5])
    targets = mx.array([1.0, 0.0, 1.0])
    
    loss_fl = focal_binary_loss(p_true, targets, gamma=2.0)
    mx.eval(loss_fl)
    
    assert float(loss_fl) >= 0.0
    
    # Easy example should have smaller loss under focal loss
    p_easy = mx.array([0.99])
    y_easy = mx.array([1.0])
    p_hard = mx.array([0.55])
    y_hard = mx.array([1.0])
    
    fl_easy = float(focal_binary_loss(p_easy, y_easy, gamma=2.0))
    fl_hard = float(focal_binary_loss(p_hard, y_hard, gamma=2.0))
    
    assert fl_easy < fl_hard

def test_brier_score_loss_properties():
    # Perfect prediction -> Brier score = 0
    p_perfect = mx.array([1.0, 0.0])
    y_perfect = mx.array([1.0, 0.0])
    
    bs_perf = float(brier_score_loss(p_perfect, y_perfect))
    assert pytest.approx(bs_perf, abs=1e-5) == 0.0

def test_combined_focal_brier_loss():
    p_true = mx.array([0.8, 0.2])
    targets = mx.array([1.0, 0.0])
    
    fl = float(focal_binary_loss(p_true, targets))
    bs = float(brier_score_loss(p_true, targets))
    comb = float(combined_focal_brier_loss(p_true, targets, brier_weight=0.5))
    
    assert pytest.approx(comb, abs=1e-5) == fl + 0.5 * bs

def test_decoupled_bool_losses():
    from core.losses import binary_cross_entropy_loss, bool_loss
    p_true = mx.array([0.8, 0.2])
    targets = mx.array([1.0, 0.0])
    
    loss_ce = bool_loss(p_true, targets, loss_type="ce")
    loss_fb = bool_loss(p_true, targets, loss_type="focal_brier")
    mx.eval(loss_ce, loss_fb)
    
    assert float(loss_ce) > 0.0
    assert float(loss_fb) > 0.0
    assert pytest.approx(float(loss_ce), abs=1e-5) == float(binary_cross_entropy_loss(p_true, targets))

def test_decoupled_choice_losses():
    from core.losses import choice_loss, choice_cross_entropy_loss
    probs = mx.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    targets = mx.array([0, 1])
    
    loss_ce = choice_loss(probs, targets, loss_type="ce")
    loss_fb = choice_loss(probs, targets, loss_type="focal_brier", brier_weight=0.5)
    mx.eval(loss_ce, loss_fb)
    
    assert float(loss_ce) > 0.0
    assert float(loss_fb) > float(loss_ce) # Brier penalty adds positive term
    assert pytest.approx(float(loss_ce), abs=1e-5) == float(choice_cross_entropy_loss(probs, targets))

def test_decoupled_score_losses():
    from core.losses import ordinal_score_loss
    logits = mx.array([[5.0, 2.0, 0.0, -1.0, -3.0]])
    target = mx.array([0])
    
    loss_one_hot = ordinal_score_loss(logits, target, num_levels=5, from_logits=True, mode="one_hot")
    loss_ordinal = ordinal_score_loss(logits, target, num_levels=5, from_logits=True, mode="ordinal_gaussian")
    mx.eval(loss_one_hot, loss_ordinal)
    
    assert float(loss_one_hot) > 0.0
    assert float(loss_ordinal) > 0.0

