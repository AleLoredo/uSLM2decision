"""
OpenJEV Cross-Model Scientific Ablation Aggregator
Consolidates individual 8-experiment ablation summaries across multiple architectures
(Qwen2.5-0.5B, OpenELM-450M, Gemma-2-2B) into a unified triangular comparison report.
Outputs:
- experiments/cross_model_comparison_table.md
- experiments/cross_model_comparison_summary.json
"""

import os
import sys
import json
import glob
from typing import Dict, List, Any

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_scanner import get_model_hidden_size

MODEL_METADATA = {
    "qwen2.5-0.5b": {
        "display_name": "Qwen 2.5-0.5B",
        "family": "Llama-like Homogeneous",
        "tier": "Fast Tier",
        "params": "0.49B",
        "layers": 24,
        "hidden_dim": 896,
        "typical_latency_ms": 61.4
    },
    "openelm-450m": {
        "display_name": "OpenELM-450M",
        "family": "Apple Layer-Varying",
        "tier": "Efficient Edge Tier",
        "params": "0.45B",
        "layers": 20,
        "hidden_dim": 1536,
        "typical_latency_ms": 93.8
    },
    "gemma-2-2b": {
        "display_name": "Gemma 2 2B",
        "family": "Google Legacy/Sliding Window",
        "tier": "Balanced Tier",
        "params": "2.00B",
        "layers": 26,
        "hidden_dim": 2304,
        "typical_latency_ms": 282.6
    }
}

def load_all_ablation_summaries(experiments_dir: str = "./experiments") -> Dict[str, List[Dict[str, Any]]]:
    summaries = {}
    
    # 1. Try experiments/ablation_summary_<model>.json
    summary_files = glob.glob(os.path.join(experiments_dir, "ablation_summary_*.json"))
    for sf in summary_files:
        basename = os.path.basename(sf)
        if basename in ("ablation_summary.json", "cross_model_comparison_summary.json"):
            continue
        model_name = basename.replace("ablation_summary_", "").replace(".json", "")
        try:
            with open(sf, "r", encoding="utf-8") as f:
                summaries[model_name] = json.load(f)
        except Exception as e:
            print(f"[Aggregator] Warning: Failed to load {sf}: {e}")

    # 2. Check experiments/run/TDS2_<model>/ablation_summary.json
    run_dir = os.path.join(experiments_dir, "run")
    if os.path.isdir(run_dir):
        for entry in os.listdir(run_dir):
            if entry.startswith("TDS2_"):
                model_name = entry.replace("TDS2_", "")
                if model_name not in summaries:
                    sub_sum = os.path.join(run_dir, entry, "ablation_summary.json")
                    if os.path.isfile(sub_sum):
                        try:
                            with open(sub_sum, "r", encoding="utf-8") as f:
                                summaries[model_name] = json.load(f)
                        except Exception as e:
                            print(f"[Aggregator] Warning: Failed to load {sub_sum}: {e}")

    return summaries

def build_triangular_comparison(summaries: Dict[str, List[Dict[str, Any]]]) -> str:
    lines = [
        "### 4.12 Cross-Architectural Triangular Synthesis & Efficiency Comparison\n",
        "A consolidated evaluation across the three distinct model lineages under identical blind test splits (`train-dataset2/test.jsonl`, $N=300$).",
        "This table compares the optimal production curriculum (**E5b: Full 3-Phase Suite with Optimizer Reset**) across all architectures:\n",
        "| Architecture Lineage | Parameter Scale | Hidden Dim ($d$) | Layers ($L$) | Bool Acc (%) | Choice Acc (%) | Score Cont. MAE | Score Severe Misses | Macro ECE (EW %) | Macro ECE (Ad %) | Macro Brier Score |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    ]

    canonical_order = ["qwen2.5-0.5b", "openelm-450m", "gemma-2-2b"]
    all_models = [m for m in canonical_order if m in summaries] + [m for m in summaries if m not in canonical_order]

    for m_name in all_models:
        exps = {r["spec"]["id"]: r for r in summaries[m_name]}
        e5b = exps.get("E5b")
        if not e5b:
            # Fallback to E6 or last
            e5b = exps.get("E6", list(exps.values())[-1])

        m = e5b["metrics"]
        meta = MODEL_METADATA.get(m_name, {
            "display_name": m_name,
            "params": "—",
            "layers": "—",
            "hidden_dim": get_model_hidden_size(f"./models/{m_name}")
        })

        score_mae = m["score"].get("mae_cont", m["score"].get("mae", 0.0))
        score_sev = m["score"].get("severe_misses_pct", 0.0)

        row = (
            f"| **{meta['display_name']}** | {meta['params']} | {meta['hidden_dim']} | {meta['layers']} | "
            f"**{m['bool']['acc']:.2f}%** | **{m['choice']['acc']:.2f}%** | **{score_mae:.3f}** | "
            f"**{score_sev:.2f}%** | {m['summary']['macro_ece_equal_width']:.2f}% | "
            f"{m['summary']['macro_ece_adaptive']:.2f}% | **{m['summary']['macro_brier']:.4f}** |"
        )
        lines.append(row)

    lines.append("\n#### Curriculum Impact: Uncalibrated Baseline (E1) vs Recommended Curriculum (E5b)\n")
    lines.append("| Architecture Lineage | E1 Bool Acc | E5b Bool Acc | E1 Choice Acc | E5b Choice Acc | E1 Score MAE | E5b Score MAE | E1 Macro ECE | E5b Macro ECE | E1 Brier | E5b Brier |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for m_name in all_models:
        exps = {r["spec"]["id"]: r for r in summaries[m_name]}
        e1 = exps.get("E1")
        e5b = exps.get("E5b")
        meta = MODEL_METADATA.get(m_name, {"display_name": m_name})

        if e1 and e5b:
            m1 = e1["metrics"]
            m5 = e5b["metrics"]
            s1_mae = m1["score"].get("mae_cont", m1["score"].get("mae", 0.0))
            s5_mae = m5["score"].get("mae_cont", m5["score"].get("mae", 0.0))
            row = (
                f"| **{meta['display_name']}** | {m1['bool']['acc']:.2f}% | **{m5['bool']['acc']:.2f}%** | "
                f"{m1['choice']['acc']:.2f}% | **{m5['choice']['acc']:.2f}%** | {s1_mae:.3f} | **{s5_mae:.3f}** | "
                f"{m1['summary']['macro_ece_equal_width']:.2f}% | **{m5['summary']['macro_ece_equal_width']:.2f}%** | "
                f"{m1['summary']['macro_brier']:.4f} | **{m5['summary']['macro_brier']:.4f}** |"
            )
            lines.append(row)

    return "\n".join(lines)

def main():
    exp_dir = "./experiments"
    summaries = load_all_ablation_summaries(exp_dir)
    print(f"[Aggregator] Loaded ablation summaries for {len(summaries)} models: {list(summaries.keys())}")

    if not summaries:
        print("[Aggregator] No ablation summaries found. Run run_ablation_matrix.py first.")
        return

    # Generate Markdown Table
    md_content = build_triangular_comparison(summaries)
    table_path = os.path.join(exp_dir, "cross_model_comparison_table.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(md_content + "\n")
    print(f"[Aggregator] Saved consolidated comparative report to: {table_path}")

    # Generate JSON summary
    summary_path = os.path.join(exp_dir, "cross_model_comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"[Aggregator] Saved consolidated JSON summary to: {summary_path}")

    print("\n" + md_content + "\n")

if __name__ == "__main__":
    main()
