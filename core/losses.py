"""
OpenJEV Ordinal & Calibrated Loss Functions (MLX Backend)
Implements training loss functions for typed decision heads per ARCHITECTURE.md §3 & §4:
1. bool_calibrated_loss: BCE + confidence calibration penalty
2. choice_cross_entropy_loss: Cross-entropy over choice option distributions
3. ordinal_score_loss: Soft-smoothed ordinal loss for score distribution (penalizing distant errors strictly more than near misses)
"""

import mlx.core as mx
import mlx.nn as nn

def bool_calibrated_loss(
    p_true: mx.array,
    targets: mx.array,
    calib_weight: float = 0.1
) -> mx.array:
    """
    Binary cross-entropy with explicit Brier / calibration regularization.
    
    Args:
        p_true: Predicted P(true), shape (B,)
        targets: Binary ground truth labels (0.0 or 1.0), shape (B,)
        calib_weight: Weight of Brier calibration penalty term
    """
    targets_float = targets.astype(mx.float32)
    bce = mx.mean(nn.losses.binary_cross_entropy(p_true, targets_float))
    brier = mx.mean((p_true - targets_float) ** 2)
    return bce + calib_weight * brier

def choice_cross_entropy_loss(
    probs: mx.array,
    target_indices: mx.array
) -> mx.array:
    """
    Cross-entropy loss for option selection.
    
    Args:
        probs: Option probability distribution (B, N)
        target_indices: Target option indices, shape (B,)
    """
    log_probs = mx.log(mx.clip(probs, 1e-9, None))
    # Gather log_probs at target indices
    batch_indices = mx.arange(probs.shape[0])
    selected_log_probs = log_probs[batch_indices, target_indices]
    return -mx.mean(selected_log_probs)

def ordinal_score_loss(
    inputs: mx.array,
    target_levels: mx.array,
    num_levels: int = 5,
    sigma: float = 0.35,
    from_logits: bool = False
) -> mx.array:
    """
    Ordinal-aware score loss using Gaussian soft-smoothed target distributions.
    Puts ground-truth mass on target level while spreading adjacent mass to nearby levels,
    ensuring distant level mistakes receive strictly higher loss penalization.
    
    Args:
        inputs: Predicted level probabilities or raw logits (B, M)
        target_levels: Ground truth level integers in [0, M-1], shape (B,)
        num_levels: Number of discrete levels M
        sigma: Soft-label smoothing bandwidth (controls ordinal distance decay)
        from_logits: Whether inputs are raw logits (numerically stable) or probabilities
    """
    levels = mx.arange(num_levels).astype(mx.float32) # (M,)
    levels_grid = mx.expand_dims(levels, 0) # (1, M)
    targets_grid = mx.expand_dims(target_levels.astype(mx.float32), -1) # (B, 1)
    
    # Distance matrix (B, M)
    distances_sq = (levels_grid - targets_grid) ** 2
    
    # Gaussian soft target distribution
    soft_targets = mx.exp(-distances_sq / (2.0 * (sigma ** 2)))
    soft_targets = soft_targets / mx.sum(soft_targets, axis=-1, keepdims=True) # Normalize
    
    # KL-Divergence / Cross-Entropy over soft ordinal targets
    if from_logits:
        # Numerically stable log_softmax: z - logsumexp(z)
        log_probs = inputs - mx.logsumexp(inputs, axis=-1, keepdims=True)
    else:
        log_probs = mx.log(mx.clip(inputs, 1e-9, None))
        
    loss = -mx.mean(mx.sum(soft_targets * log_probs, axis=-1))
    
    return loss

