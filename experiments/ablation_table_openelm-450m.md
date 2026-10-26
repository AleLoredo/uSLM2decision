### Ablation Study: Training Methodology & Decoupled Calibration Matrix (openelm-450m)

| Experiment ID | Pipeline Configuration | Fase 2 Opt | Bool Acc (%) | Choice Acc (%) | Score Cont MAE | Score Severe (>1.5) | ECE (%) [Equal Width] | ECE (%) [Adaptive] | Brier Score |
|---|---|---|---|---|---|---|---|---|---|
| **E1** | Phase 1 (Classical CE) | — | 61.00 | 26.00 | 0.654 | 1.00% | 6.92 | 13.76 | 0.6204 |
| **E2** | Phase 1 (CE) + Phase 3 (Post-Hoc T) | — | 61.00 | 26.00 | 0.654 | 1.00% | 6.27 | 14.22 | 0.6227 |
| **E3** | Phase 1 (Focal + Brier) | — | 54.00 | 28.00 | 0.654 | 1.00% | 13.85 | 16.42 | 0.6289 |
| **E4a** | Phase 1 (CE) + Phase 2 (Refinement) | Continuous | 64.00 | 26.00 | 0.640 | 1.00% | 7.11 | 16.21 | 0.6211 |
| **E4b** | Phase 1 (CE) + Phase 2 (Refinement) | Reset | 64.00 | 24.00 | 0.629 | 1.00% | 9.62 | 16.44 | 0.6126 |
| **E5a** | Full Curriculum (P1 + P2 + P3) | Continuous | 64.00 | 26.00 | 0.640 | 1.00% | 8.61 | 16.10 | 0.6290 |
| **E5b** | Full Curriculum (P1 + P2 + P3) | Reset | 64.00 | 24.00 | 0.629 | 1.00% | 10.25 | 17.56 | 0.6228 |
| **E6** | Direct Pipeline (P1[FB] + P3) | — | 54.00 | 28.00 | 0.654 | 1.00% | 6.82 | 17.50 | 0.6306 |
