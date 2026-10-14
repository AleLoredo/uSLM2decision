"""
OpenJEV Normative Evaluation Metrics
Implements requirements from:
- SPEC.md §8 (Calibration Reporting Requirements)
- SPEC.md §5 (Abstention & AURC)
- ARCHITECTURE.md §5 (Evaluation Metrics: ECE, Brier Decomposition, AURC, RPS)
"""

from typing import Literal
import numpy as np

def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    scheme: Literal["equal_width", "adaptive"] = "equal_width"
) -> dict:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    
    Args:
        probs: Array of predicted probabilities for the chosen class, shape (N,)
               or full multiclass distribution, shape (N, K).
        labels: Ground truth binary (N,) or one-hot (N, K) or true class indices (N,).
        n_bins: Number of bins (e.g. 10 or 15).
        scheme: 'equal_width' (fixed interval bins) or 'adaptive' (quantile equal-count bins).
    
    Returns:
        dict with 'ece', 'mce', 'bin_accuracies', 'bin_confidences', 'bin_counts', 'scheme'
    """
    # Normalize inputs to top-1 predictions if multiclass
    if probs.ndim == 2:
        confidences = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        if labels.ndim == 2:
            true_labels = np.argmax(labels, axis=1)
        else:
            true_labels = labels
        accuracies = (preds == true_labels).astype(float)
    else:
        confidences = np.asarray(probs, dtype=float)
        accuracies = np.asarray(labels, dtype=float)

    n_samples = len(confidences)
    if n_samples == 0:
        return {"ece": 0.0, "mce": 0.0, "scheme": scheme}

    if scheme == "equal_width":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_assignments = np.digitize(confidences, bin_edges) - 1
        bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)
    elif scheme == "adaptive":
        # Quantile binning: equal number of samples per bin
        quantiles = np.linspace(0.0, 100.0, n_bins + 1)
        bin_edges = np.percentile(confidences, quantiles)
        bin_assignments = np.digitize(confidences, bin_edges, right=True) - 1
        bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    ece = 0.0
    mce = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for b in range(n_bins):
        mask = bin_assignments == b
        count = int(np.sum(mask))
        bin_counts.append(count)
        if count > 0:
            avg_acc = float(np.mean(accuracies[mask]))
            avg_conf = float(np.mean(confidences[mask]))
            diff = abs(avg_acc - avg_conf)
            ece += (count / n_samples) * diff
            mce = max(mce, diff)
            bin_accs.append(avg_acc)
            bin_confs.append(avg_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
        "n_bins": n_bins,
        "scheme": scheme
    }

def compute_brier_decomposition(probs: np.ndarray, labels: np.ndarray) -> dict:
    """
    Computes Brier Score and Murphy's Brier Score Decomposition:
    Brier = Reliability - Resolution + Uncertainty
    
    Args:
        probs: Array of predicted probabilities (N, K) or (N,)
        labels: One-hot ground truth (N, K) or binary (N,)
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    
    if probs.ndim == 1:
        probs = np.stack([1.0 - probs, probs], axis=1)
        labels = np.stack([1.0 - labels, labels], axis=1)
    elif labels.ndim == 1 and probs.ndim == 2:
        k = probs.shape[1]
        labels = np.eye(k)[labels.astype(int)]
        
    n_samples, n_classes = probs.shape
    # Raw Brier Score
    brier_score = float(np.mean(np.sum((probs - labels) ** 2, axis=1)))
    
    # Base rate / Climatology / Base uncertainty
    base_rates = np.mean(labels, axis=0) # [K]
    uncertainty = float(np.sum(base_rates * (1.0 - base_rates)))
    
    # Stratify by predicted probability bins to compute reliability and resolution
    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    
    reliability = 0.0
    resolution = 0.0
    
    for k in range(n_classes):
        p_k = probs[:, k]
        y_k = labels[:, k]
        base_rate_k = base_rates[k]
        
        bin_assignments = np.clip(np.digitize(p_k, bin_edges) - 1, 0, n_bins - 1)
        for b in range(n_bins):
            mask = bin_assignments == b
            count = int(np.sum(mask))
            if count > 0:
                bar_p_kb = float(np.mean(p_k[mask]))
                bar_y_kb = float(np.mean(y_k[mask]))
                weight = count / n_samples
                
                # Reliability component: sum (bar_p - bar_y)^2
                reliability += weight * ((bar_p_kb - bar_y_kb) ** 2)
                # Resolution component: sum (bar_y - base_rate)^2
                resolution += weight * ((bar_y_kb - base_rate_k) ** 2)

    return {
        "brier_score": brier_score,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": uncertainty,
        "brier_check": float(reliability - resolution + uncertainty)
    }

def compute_aurc(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_thresholds: int = 100
) -> dict:
    """
    Computes Area Under Risk-Coverage curve (AURC) for selective prediction/abstention (SPEC.md §5).
    
    Risk = Error Rate on accepted predictions.
    Coverage = Fraction of accepted predictions.
    """
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=bool)
    n_samples = len(confidences)
    
    sorted_indices = np.argsort(-confidences) # Descending confidence
    sorted_correctness = correctness[sorted_indices]
    
    coverages = []
    risks = []
    
    cum_correct = np.cumsum(sorted_correctness)
    for i in range(1, n_samples + 1):
        coverage = i / n_samples
        accuracy = cum_correct[i - 1] / i
        risk = 1.0 - accuracy
        coverages.append(coverage)
        risks.append(risk)
        
    aurc = float(np.trapezoid(risks, coverages)) if hasattr(np, 'trapezoid') else float(np.trapz(risks, coverages))
    
    return {
        "aurc": aurc,
        "coverages": coverages[::max(1, len(coverages) // n_thresholds)],
        "risks": risks[::max(1, len(risks) // n_thresholds)]
    }

def compute_ranked_probability_score(probs: np.ndarray, true_levels: np.ndarray) -> dict:
    """
    Computes Ranked Probability Score (RPS) for ordinal score distributions (SPEC.md §3.3 & ARCHITECTURE.md §5).
    
    RPS = (1 / (M - 1)) sum_{m=1}^{M-1} (CDF_pred(m) - CDF_true(m))^2
    """
    probs = np.asarray(probs, dtype=float)
    true_levels = np.asarray(true_levels, dtype=int)
    
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
        true_levels = true_levels.reshape(1)
        
    n_samples, n_levels = probs.shape
    rps_list = []
    
    for i in range(n_samples):
        pred_cdf = np.cumsum(probs[i])
        true_cdf = np.zeros(n_levels)
        true_cdf[true_levels[i]:] = 1.0
        
        # Sum of squared differences over CDF bins, normalized by (M - 1)
        sample_rps = float(np.sum((pred_cdf[:-1] - true_cdf[:-1]) ** 2) / (n_levels - 1))
        rps_list.append(sample_rps)
        
    return {
        "mean_rps": float(np.mean(rps_list)),
        "individual_rps": rps_list
    }
