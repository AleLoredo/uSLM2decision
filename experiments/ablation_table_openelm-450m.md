### Ablation Study: Training Methodology & Decoupled Calibration Matrix (openelm-450m)

| Experiment ID | Pipeline Configuration | Fase 2 Opt | Bool F1 (%) [P] | Bool Acc (%) [S] | Choice Top-1 (%) [P] | Score Cont MAE | Score Severe (>1.5) | ECE (%) [Equal Width] | ECE (%) [Adaptive] | Brier Score |
|---|---|---|---|---|---|---|---|---|---|---|
| **E1** | Phase 1 (Classical CE) | — | 99.01 | 99.00 | 97.00 | 0.260 | 0.00% | 8.01 | 8.01 | 0.0617 |
| **E2** | Phase 1 (CE) + Phase 3 (Post-Hoc T) | — | 99.01 | 99.00 | 97.00 | 0.260 | 0.00% | 5.97 | 5.97 | 0.0544 |
| **E3** | Phase 1 (Focal + Brier) | — | 100.00 | 100.00 | 96.00 | 0.265 | 0.00% | 9.11 | 9.11 | 0.0718 |
| **E4a** | Phase 1 (CE) + Phase 2 (Refinement) | Continuous | 99.01 | 99.00 | 97.00 | 0.254 | 0.00% | 6.98 | 7.44 | 0.0542 |
| **E4b** | Phase 1 (CE) + Phase 2 (Refinement) | Reset | 100.00 | 100.00 | 98.00 | 0.216 | 0.00% | 4.20 | 3.94 | 0.0306 |
| **E5a** | Full Curriculum (P1 + P2 + P3) | Continuous | 99.01 | 99.00 | 97.00 | 0.254 | 0.00% | 4.72 | 5.18 | 0.0449 |
| **E5b** | Full Curriculum (P1 + P2 + P3) | Reset | 100.00 | 100.00 | 98.00 | 0.216 | 0.00% | 1.97 | 1.70 | 0.0233 |
| **E6** | Direct Pipeline (P1[FB] + P3) | — | 100.00 | 100.00 | 96.00 | 0.265 | 0.00% | 5.35 | 5.35 | 0.0521 |
