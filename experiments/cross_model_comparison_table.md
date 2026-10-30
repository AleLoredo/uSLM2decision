### 4.12 Cross-Architectural Triangular Synthesis & Efficiency Comparison

A consolidated evaluation across the three distinct model lineages under identical blind test splits (`train-dataset2/test.jsonl`, $N=300$).
This table compares the optimal production curriculum (**E5b: Full 3-Phase Suite with Optimizer Reset**) across all architectures:

| Architecture Lineage | Parameter Scale | Hidden Dim ($d$) | Layers ($L$) | Bool F1 (%) [P] | Bool Acc (%) [S] | Choice Top-1 (%) [P] | Score Cont. MAE | Score Severe Misses | Macro ECE (EW %) | Macro ECE (Ad %) | Macro Brier Score |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen 2.5-0.5B** | 0.49B | 896 | 24 | **96.08%** | 96.00% | **96.00%** | **0.389** | **0.00%** | 5.27% | 5.47% | **0.0780** |
| **OpenELM-450M** | 0.45B | 1536 | 20 | **100.00%** | 100.00% | **98.00%** | **0.216** | **0.00%** | 1.97% | 1.70% | **0.0233** |
| **Gemma 2 2B** | 2.00B | 2304 | 26 | **91.67%** | 92.00% | **96.00%** | **0.383** | **0.00%** | 7.34% | 5.02% | **0.1082** |

#### Curriculum Impact: Uncalibrated Baseline (E1) vs Recommended Curriculum (E5b)

| Architecture Lineage | E1 Bool F1 [P] | E5b Bool F1 [P] | E1 Bool Acc [S] | E5b Bool Acc [S] | E1 Choice Top-1 [P] | E5b Choice Top-1 [P] | E1 Score MAE | E5b Score MAE | E1 Macro ECE | E5b Macro ECE | E1 Brier | E5b Brier |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen 2.5-0.5B** | 93.33% | **96.08%** | 93.00% | **96.00%** | 79.00% | **96.00%** | 0.432 | **0.389** | 15.25% | **5.27%** | 0.2245 | **0.0780** |
| **OpenELM-450M** | 99.01% | **100.00%** | 99.00% | **100.00%** | 97.00% | **98.00%** | 0.260 | **0.216** | 8.01% | **1.97%** | 0.0617 | **0.0233** |
| **Gemma 2 2B** | 89.36% | **91.67%** | 90.00% | **92.00%** | 74.00% | **96.00%** | 0.411 | **0.383** | 12.65% | **7.34%** | 0.2692 | **0.1082** |
