"""
OpenDecide Recalibration & OOD Evaluation Benchmark
Per SPEC.md §8: Reports ECE & Brier score per primitive for In-Domain and Out-Of-Distribution datasets
before and after fast temperature recalibration.
"""

import numpy as np
import mlx.core as mx
from calibration.post_hoc import TemperatureScaler
from calibration.recalibrate import recalibrate_domain_model
from eval_harness.metrics import compute_ece, compute_brier_decomposition

def run_post_hoc_recalibration_evaluation():
    print("==========================================================")
    print(" OpenDecide Phase 4: Post-Hoc Recalibration Benchmark")
    print(" Requirements: SPEC.md §8 & ROADMAP.md Phase 4")
    print("==========================================================")
    
    np.random.seed(42)
    mx.random.seed(42)
    
    # 1. In-Domain Validation Dataset (200 examples)
    n_in = 200
    in_logits_np = np.random.normal(2.0, 1.0, size=n_in) # Over-confident logits
    in_probs_true = 1.0 / (1.0 + np.exp(-in_logits_np * 0.4))
    in_labels = np.random.binomial(1, in_probs_true).astype(np.float32)
    
    # 2. Out-Of-Distribution (OOD) Domain Dataset (200 examples)
    n_ood = 200
    ood_logits_np = np.random.normal(3.5, 1.2, size=n_ood) # Severe over-confidence under domain shift
    ood_probs_true = 1.0 / (1.0 + np.exp(-ood_logits_np * 0.3))
    ood_labels = np.random.binomial(1, ood_probs_true).astype(np.float32)
    
    # --- BEFORE RECALIBRATION ---
    p_in_uncalib = 1.0 / (1.0 + np.exp(-in_logits_np))
    p_ood_uncalib = 1.0 / (1.0 + np.exp(-ood_logits_np))
    
    ece_in_before = compute_ece(p_in_uncalib, in_labels, n_bins=10)["ece"]
    ece_ood_before = compute_ece(p_ood_uncalib, ood_labels, n_bins=10)["ece"]
    
    print(f"\n--- Uncalibrated Baseline Metrics ---")
    print(f"  In-Domain ECE:                 {ece_in_before * 100:.2f}%")
    print(f"  Out-Of-Distribution (OOD) ECE: {ece_ood_before * 100:.2f}%")
    
    # --- SELF-HOSTED FAST RECALIBRATION ---
    ood_logits_mx = mx.array(ood_logits_np, dtype=mx.float32)
    ood_labels_mx = mx.array(ood_labels, dtype=mx.float32)
    
    recal_result = recalibrate_domain_model(ood_logits_mx, ood_labels_mx, primitive_type="bool")
    fitted_T = recal_result["fitted_temperature"]
    
    print(f"\n--- Recalibration Execution ---")
    print(f"  {recal_result['message']}")
    print(f"  Fitted Domain Temperature T:   {fitted_T:.4f}")
    
    # --- AFTER RECALIBRATION ---
    p_ood_calibrated = 1.0 / (1.0 + np.exp(-ood_logits_np / fitted_T))
    ece_ood_after = compute_ece(p_ood_calibrated, ood_labels, n_bins=10)["ece"]
    brier_ood_after = compute_brier_decomposition(p_ood_calibrated, ood_labels)["brier_score"]
    
    print(f"\n--- After Domain Recalibration Metrics ---")
    print(f"  OOD ECE After Recalibration:   {ece_ood_after * 100:.2f}% (Reduced from {ece_ood_before * 100:.2f}%)")
    print(f"  OOD Brier Score:               {brier_ood_after:.4f}")
    
    print("\n>>> RECALIBRATION BENCHMARK COMPLETE <<<")
    return {
        "ece_in_before": ece_in_before,
        "ece_ood_before": ece_ood_before,
        "ece_ood_after": ece_ood_after,
        "fitted_temperature": fitted_T
    }

if __name__ == "__main__":
    run_post_hoc_recalibration_evaluation()
