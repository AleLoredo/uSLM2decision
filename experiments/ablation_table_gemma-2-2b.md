### Ablation Study: Training Methodology & Decoupled Calibration Matrix (gemma-2-2b)

| Experiment ID | Pipeline Configuration | Fase 2 Opt | Bool Acc (%) | Choice Acc (%) | Score Cont MAE | Score Severe (>1.5) | ECE (%) [Equal Width] | ECE (%) [Adaptive] | Brier Score |
|---|---|---|---|---|---|---|---|---|---|
| **E1** | Phase 1 (Classical CE) | — | 90.00 | 74.00 | 0.411 | 0.00% | 12.65 | 12.14 | 0.2692 |
| **E2** | Phase 1 (CE) + Phase 3 (Post-Hoc T) | — | 90.00 | 74.00 | 0.411 | 0.00% | 12.02 | 12.19 | 0.2697 |
| **E3** | Phase 1 (Focal + Brier) | — | 50.00 | 82.00 | 0.409 | 0.00% | 33.99 | 33.99 | 0.6551 |
| **E4a** | Phase 1 (CE) + Phase 2 (Refinement) | Continuous | 91.00 | 83.00 | 0.402 | 0.00% | 13.19 | 11.42 | 0.2177 |
| **E4b** | Phase 1 (CE) + Phase 2 (Refinement) | Reset | 92.00 | 96.00 | 0.383 | 0.00% | 6.60 | 5.34 | 0.1087 |
| **E5a** | Full Curriculum (P1 + P2 + P3) | Continuous | 91.00 | 83.00 | 0.402 | 0.00% | 12.95 | 11.29 | 0.2173 |
| **E5b** | Full Curriculum (P1 + P2 + P3) | Reset | 92.00 | 96.00 | 0.383 | 0.00% | 7.34 | 5.02 | 0.1082 |
| **E6** | Direct Pipeline (P1[FB] + P3) | — | 50.00 | 82.00 | 0.409 | 0.00% | 33.99 | 33.99 | 0.6551 |
