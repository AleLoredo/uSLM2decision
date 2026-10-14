"""
OpenJEV Post-Hoc Calibration Engine (MLX Backend)
Implements temperature scaling and vector scaling calibration for decision logits.
Per SPEC.md §8 & ARCHITECTURE.md §4.
"""

import math
from typing import Dict, Tuple, Optional
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from calibration.metrics import compute_ece

class TemperatureScaler:
    """
    Fits a single temperature scalar T > 0 on validation logits using Grid Search on ECE / Brier Score.
    p_calibrated = softmax(logits / T) or sigmoid(logits / T)
    """
    def __init__(self, initial_temperature: float = 1.0):
        self.temperature = initial_temperature

    def fit(self, logits: mx.array, targets: mx.array, search_range: Tuple[float, float] = (0.5, 5.0), steps: int = 150) -> float:
        """
        Finds optimal temperature T in search_range that minimizes Empirical Calibration Error (ECE).
        
        Args:
            logits: Unscaled logit predictions (B,) or (B, N)
            targets: Target binary values or option indices
            search_range: Range of temperatures (min_T, max_T) to search
            steps: Number of evaluation grid steps
        """
        targets_np = np.array(targets)
        logits_np = np.array(logits)
        
        candidates = np.linspace(search_range[0], search_range[1], steps)
        
        best_t = self.temperature
        best_ece = float('inf')
        
        for t_val in candidates:
            if logits.ndim == 1:
                p_calib = 1.0 / (1.0 + np.exp(-logits_np / t_val))
            else:
                exps = np.exp(logits_np / t_val)
                p_calib = exps / np.sum(exps, axis=-1, keepdims=True)
                
            ece_res = compute_ece(p_calib, targets_np, n_bins=10)
            ece = ece_res["ece"]
            
            if ece < best_ece:
                best_ece = ece
                best_t = t_val
                
        self.temperature = float(best_t)
        return self.temperature

    def scale_logits(self, logits: mx.array) -> mx.array:
        """
        Applies learned temperature scaling to input logits.
        """
        return logits / self.temperature
