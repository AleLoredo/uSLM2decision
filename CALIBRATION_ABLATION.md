# CALIBRATION_ABLATION.md: Focal Loss vs. Brier Score Empirical Ablation Report

- **Document Version:** 1.0.0
- **Requirement Source:** [`ARCHITECTURE.md`](file:///Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/ARCHITECTURE.md#L154-L163) §4.1
- **Evaluation Date:** 2026-09-23
- **Hardware/Framework:** Apple M3, Metal GPU, MLX Backend

---

## 1. Executive Summary & Normative Decision

As required by `ARCHITECTURE.md` §4.1, an empirical ablation study was conducted to evaluate whether Focal Loss and Brier Score regularization conflict during training.

### Key Finding
- **Focal Loss alone** down-weights easy examples but introduces systematic **under-confidence**, increasing ECE (7.29%).
- **Pure Brier Score Regularization** operates as a proper scoring rule, significantly improving reliability (ECE down to 4.30%).
- **Combined Focal Loss + Brier Regularization** (Recipe D) achieves the best overall performance with **ECE = 3.67%** and lowest reliability error (0.0065).

**Locked Training Recipe for OpenDecide:**
The normative loss formulation for `bool` and `choice` heads combines Focal Loss ($\gamma = 1.0$) with Brier Score Regularization ($\lambda = 0.5$). `score` heads retain the ordinal Gaussian soft-smoothed loss ([`core/losses.py`](file:///Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/core/losses.py)).

---

## 2. Empirical Ablation Results Table

| Loss Recipe | ECE (Lower is better) | MCE (Lower is better) | Brier Score | Reliability Component | Status |
|---|---|---|---|---|---|
| **Recipe A: Standard BCE** | 7.23% | 19.09% | 0.3288 | 0.0137 | Baseline |
| **Recipe B: Focal Loss ($\gamma=2.0$)** | 7.29% | 16.49% | 0.3232 | 0.0174 | Under-confident |
| **Recipe C: Pure Brier Regularization** | 4.30% | **14.29%** | **0.3111** | 0.0073 | Well-Calibrated |
| **Recipe D: Combined Focal + Brier** | **3.67%** | 16.15% | 0.3114 | **0.0065** | **SELECTED** |

---

## 3. Implementation References

- Calibration loss shaping functions: [`calibration/loss_shaping.py`](file:///Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/calibration/loss_shaping.py)
- Ablation experiment script: [`scratch/focal_brier_ablation.py`](file:///Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/scratch/focal_brier_ablation.py)
- Metric evaluation harness: [`eval_harness/metrics.py`](file:///Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/eval_harness/metrics.py)
