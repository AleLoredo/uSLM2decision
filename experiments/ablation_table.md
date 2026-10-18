### 4.9 Ablation Study: Training Methodology & Decoupled Calibration Matrix

| Experiment ID | Pipeline Configuration | Fase 2 Opt | Bool Acc (%) | Choice Acc (%) | Score Acc (%) | ECE (%) [Equal Width] | ECE (%) [Adaptive] | Brier Score | Score Severe Misses |
|---|---|---|---|---|---|---|---|---|---|
| **E1** | Phase 1 (Classical CE) | — | 97.14 | 98.31 | 100.00 | 4.42 | 4.00 | 0.0274 | 0.00% |
| **E2** | Phase 1 (CE) + Phase 3 (Post-Hoc T) | — | 97.14 | 98.31 | 100.00 | 3.39 | 3.25 | 0.0297 | 0.00% |
| **E3** | Phase 1 (Focal + Brier) | — | 98.57 | 98.31 | 100.00 | 4.76 | 3.54 | 0.0190 | 0.00% |
| **E4a** | Phase 1 (CE) + Phase 2 (Refinement) | Continuous | 97.14 | 100.00 | 100.00 | 4.24 | 4.01 | 0.0230 | 0.00% |
| **E4b** | Phase 1 (CE) + Phase 2 (Refinement) | Reset | 98.57 | 100.00 | 98.04 | 2.01 | 1.91 | 0.0091 | 0.00% |
| **E5a** | Full Curriculum (P1 + P2 + P3) | Continuous | 97.14 | 100.00 | 100.00 | 3.25 | 3.26 | 0.0252 | 0.00% |
| **E5b** | Full Curriculum (P1 + P2 + P3) | Reset | 98.57 | 100.00 | 98.04 | 1.11 | 1.10 | 0.0090 | 0.00% |
| **E6** | Direct Pipeline (P1[FB] + P3) | — | 98.57 | 98.31 | 100.00 | 3.20 | 2.29 | 0.0168 | 0.00% |
