"""
OpenJEV Ordinal & Calibrated Loss Functions (MLX Backend)
Implements training loss functions for typed decision heads per ARCHITECTURE.md §3 & §4:
1. binary_cross_entropy_loss: Pure unregularized BCE for baseline experiments
2. bool_loss / bool_calibrated_loss: Loss selector (BCE vs Focal + Brier)
3. choice_loss / choice_cross_entropy_loss: Pure CE vs CE + Multiclass Brier Regularization
4. ordinal_score_loss: Pure one-hot categorical CE vs soft-smoothed ordinal Gaussian loss
"""

import mlx.core as mx
import mlx.nn as nn
from calibration.loss_shaping import combined_focal_brier_loss, focal_binary_loss, brier_score_loss

def binary_cross_entropy_loss(p_true: mx.array, targets: mx.array) -> mx.array:
    """
    Standard Binary Cross-Entropy without regularization.
    L_BCE = -mean(y * log(p) + (1-y) * log(1-p))
    """
    p_clamped = mx.clip(p_true, 1e-7, 1.0 - 1e-7)
    targets_float = targets.astype(mx.float32)
    bce = -(targets_float * mx.log(p_clamped) + (1.0 - targets_float) * mx.log(1.0 - p_clamped))
    return mx.mean(bce)

def bool_loss(
    p_true: mx.array,
    targets: mx.array,
    loss_type: str = "focal_brier",
    gamma: float = 1.0,
    brier_weight: float = 0.5
) -> mx.array:
    """
    Configurable loss for BoolHead.
    - 'ce': Standard unregularized BCE
    - 'focal_brier': Combined Focal Loss + Brier Score Penalty
    """
    if loss_type == "ce":
        return binary_cross_entropy_loss(p_true, targets)
    elif loss_type == "focal_brier":
        return combined_focal_brier_loss(p_true, targets, gamma=gamma, brier_weight=brier_weight)
    else:
        raise ValueError(f"Unknown bool loss_type: {loss_type}")

def bool_calibrated_loss(
    p_true: mx.array,
    targets: mx.array,
    calib_weight: float = 0.1
) -> mx.array:
    """
    Binary cross-entropy with explicit Brier calibration regularization.
    """
    targets_float = targets.astype(mx.float32)
    bce = binary_cross_entropy_loss(p_true, targets_float)
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
    batch_indices = mx.arange(probs.shape[0])
    selected_log_probs = log_probs[batch_indices, target_indices]
    return -mx.mean(selected_log_probs)

def choice_loss(
    probs: mx.array,
    target_indices: mx.array,
    loss_type: str = "focal_brier",
    brier_weight: float = 0.5
) -> mx.array:
    """
    Configurable loss for ChoiceHead.
    - 'ce': Standard categorical Softmax Cross-Entropy
    - 'focal_brier': Categorical Cross-Entropy + Multiclass Brier score penalty
    """
    ce = choice_cross_entropy_loss(probs, target_indices)
    if loss_type == "ce":
        return ce
    elif loss_type == "focal_brier":
        n_classes = probs.shape[1]
        one_hot = mx.eye(n_classes)[target_indices]
        brier = mx.mean(mx.sum((probs - one_hot) ** 2, axis=-1))
        return ce + brier_weight * brier
    else:
        raise ValueError(f"Unknown choice loss_type: {loss_type}")

def ordinal_score_loss(
    inputs: mx.array,
    target_levels: mx.array,
    num_levels: int = 5,
    sigma: float = 0.35,
    from_logits: bool = False,
    mode: str = "ordinal_gaussian"
) -> mx.array:
    """
    Score loss supporting both baseline one-hot CE and ordinal-aware Gaussian smoothing.
    
    Args:
        inputs: Predicted level probabilities or raw logits (B, M)
        target_levels: Ground truth level integers in [0, M-1], shape (B,)
        num_levels: Number of discrete levels M
        sigma: Soft-label smoothing bandwidth (controls ordinal distance decay)
        from_logits: Whether inputs are raw logits or probabilities
        mode: 'ordinal_gaussian' (ordinal distance penalty) or 'one_hot' (standard categorical CE)
    """
    if mode == "one_hot":
        if from_logits:
            log_probs = inputs - mx.logsumexp(inputs, axis=-1, keepdims=True)
        else:
            log_probs = mx.log(mx.clip(inputs, 1e-9, None))
        batch_indices = mx.arange(inputs.shape[0])
        selected_log_probs = log_probs[batch_indices, target_levels]
        return -mx.mean(selected_log_probs)
    elif mode == "ordinal_gaussian":
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
            log_probs = inputs - mx.logsumexp(inputs, axis=-1, keepdims=True)
        else:
            log_probs = mx.log(mx.clip(inputs, 1e-9, None))
            
        loss = -mx.mean(mx.sum(soft_targets * log_probs, axis=-1))
        return loss
    else:
        raise ValueError(f"Unknown score loss mode: {mode}")


