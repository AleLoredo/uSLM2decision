### 4.9 Ablation Study: Training Methodology & Decoupled Calibration Matrix

| Experiment ID | Pipeline Configuration | Fase 2 Opt | Bool Acc (%) | Choice Acc (%) | Score Cont MAE | Score Severe (>1.5) | ECE (%) [Equal Width] | ECE (%) [Adaptive] | Brier Score |
|---|---|---|---|---|---|---|---|---|---|
| **E1** | Phase 1 (Classical CE) | — | 93.00 | 79.00 | 0.432 | 2.00% | 15.25 | 13.98 | 0.2245 |
| **E2** | Phase 1 (CE) + Phase 3 (Post-Hoc T) | — | 93.00 | 79.00 | 0.432 | 2.00% | 15.00 | 13.40 | 0.2206 |
| **E3** | Phase 1 (Focal + Brier) | — | 94.00 | 72.00 | 0.437 | 2.00% | 13.27 | 13.93 | 0.2417 |
| **E4a** | Phase 1 (CE) + Phase 2 (Refinement) | Continuous | 96.00 | 85.00 | 0.411 | 1.00% | 11.28 | 10.10 | 0.1635 |
| **E4b** | Phase 1 (CE) + Phase 2 (Refinement) | Reset | 96.00 | 96.00 | 0.389 | 0.00% | 7.04 | 6.07 | 0.0789 |
| **E5a** | Full Curriculum (P1 + P2 + P3) | Continuous | 96.00 | 85.00 | 0.411 | 1.00% | 10.33 | 9.27 | 0.1614 |
| **E5b** | Full Curriculum (P1 + P2 + P3) | Reset | 96.00 | 96.00 | 0.389 | 0.00% | 5.27 | 5.47 | 0.0780 |
| **E6** | Direct Pipeline (P1[FB] + P3) | — | 94.00 | 72.00 | 0.437 | 2.00% | 12.79 | 13.41 | 0.2399 |
