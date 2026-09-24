"""
OpenJEV Calibration Loss Shaping Module (MLX Backend)
Implements calibration-aware loss functions per ARCHITECTURE.md §4:
1. Focal Loss (for hard boundary example focus)
2. Brier Score Regularization (proper scoring rule)
3. Combined Focal + Brier Loss
"""

import mlx.core as mx
import mlx.nn as nn

def focal_binary_loss(
    p_true: mx.array,
    targets: mx.array,
    gamma: float = 2.0,
    alpha: float = 0.25
) -> mx.array:
    """
    Binary Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        p_true: Predicted P(true), shape (B,)
        targets: Binary ground truth labels (0.0 or 1.0), shape (B,)
        gamma: Focusing parameter (gamma > 0 down-weights easy examples)
        alpha: Class weighting factor
    """
    p_true_clamped = mx.clip(p_true, 1e-7, 1.0 - 1e-7)
    targets_float = targets.astype(mx.float32)
    
    # p_t: probability of true class
    p_t = targets_float * p_true_clamped + (1.0 - targets_float) * (1.0 - p_true_clamped)
    alpha_t = targets_float * alpha + (1.0 - targets_float) * (1.0 - alpha)
    
    focal_weight = alpha_t * ((1.0 - p_t) ** gamma)
    loss = -focal_weight * mx.log(p_t)
    return mx.mean(loss)

def brier_score_loss(
    p_true: mx.array,
    targets: mx.array
) -> mx.array:
    """
    Brier Score Loss (Proper Scoring Rule): BS = mean((p - y)^2)
    """
    targets_float = targets.astype(mx.float32)
    return mx.mean((p_true - targets_float) ** 2)

def combined_focal_brier_loss(
    p_true: mx.array,
    targets: mx.array,
    gamma: float = 2.0,
    brier_weight: float = 0.5
) -> mx.array:
    """
    Combined Focal Loss + Brier Regularization
    """
    fl = focal_binary_loss(p_true, targets, gamma=gamma)
    bs = brier_score_loss(p_true, targets)
    return fl + brier_weight * bs
