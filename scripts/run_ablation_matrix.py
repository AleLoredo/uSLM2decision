"""
OpenJEV Automated Scientific Ablation Matrix Runner
Executes the 8 canonical decoupled training experiments (E1 to E6) on Apple Silicon Metal GPU:
- E1:  Vanilla Baseline (CE, T=1.0)
- E2:  Post-Hoc Only (CE + Post-Hoc T)
- E3:  Intrinsic Only (Scratch Focal+Brier, T=1.0)
- E4a: Warm-Start Refinement with Continuous AdamW Momentum (T=1.0)
- E4b: Warm-Start Refinement with Reset AdamW Optimizer (T=1.0)
- E5a: Full Curriculum Pipeline with Continuous Momentum (P1 + P2 + P3)
- E5b: Full Curriculum Pipeline with Reset Optimizer (P1 + P2 + P3)
- E6:  Direct Production Pipeline (P1[FB] + P3)

Extracts backbone features once into memory for zero redundant inference passes,
runs the suite in seconds, and outputs a formatted Markdown table for research/paper.md.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx_lm
from scripts.train_heads import HeadTrainer

EXPERIMENT_SPECS = [
    {
        "id": "E1",
        "name": "Vanilla Baseline",
        "pipeline_label": "Phase 1 (Classical CE)",
        "phase1_loss": "ce",
        "phase1_epochs": 60,
        "phase1_lr": 1e-3,
        "enable_phase2": False,
        "phase2_epochs": 0,
        "phase2_lr": 0.0,
        "phase2_optimizer": "—",
        "skip_phase3": True,
        "desc": "Mide la sobreconfianza cruda del LLM sin ninguna técnica de calibración (T=1.0)."
    },
    {
        "id": "E2",
        "name": "Post-Hoc Only",
        "pipeline_label": "Phase 1 (CE) + Phase 3 (Post-Hoc T)",
        "phase1_loss": "ce",
        "phase1_epochs": 60,
        "phase1_lr": 1e-3,
        "enable_phase2": False,
        "phase2_epochs": 0,
        "phase2_lr": 0.0,
        "phase2_optimizer": "—",
        "skip_phase3": False,
        "desc": "Evalúa el enfoque tradicional de la literatura (Guo et al., 2017) sobre Cross-Entropy."
    },
    {
        "id": "E3",
        "name": "Intrinsic Only (Scratch)",
        "pipeline_label": "Phase 1 (Focal + Brier)",
        "phase1_loss": "focal_brier",
        "phase1_epochs": 60,
        "phase1_lr": 1e-3,
        "enable_phase2": False,
        "phase2_epochs": 0,
        "phase2_lr": 0.0,
        "phase2_optimizer": "—",
        "skip_phase3": True,
        "desc": "Evalúa cuánto logra la regularización Brier por sí sola sin ajuste post-hoc (T=1.0)."
    },
    {
        "id": "E4a",
        "name": "Warm-Start (Continuous)",
        "pipeline_label": "Phase 1 (CE) + Phase 2 (Refinement)",
        "phase1_loss": "ce",
        "phase1_epochs": 60,
        "phase1_lr": 1e-3,
        "enable_phase2": True,
        "phase2_epochs": 30,
        "phase2_lr": 2e-4,
        "phase2_optimizer": "continuous",
        "skip_phase3": True,
        "desc": "Mide el efecto de retener el momentum de AdamW al cambiar de función objetivo."
    },
    {
        "id": "E4b",
        "name": "Warm-Start (Reset Opt)",
        "pipeline_label": "Phase 1 (CE) + Phase 2 (Refinement)",
        "phase1_loss": "ce",
        "phase1_epochs": 60,
        "phase1_lr": 1e-3,
        "enable_phase2": True,
        "phase2_epochs": 30,
        "phase2_lr": 2e-4,
        "phase2_optimizer": "reset",
        "skip_phase3": True,
        "desc": "Mide si re-inicializar el momentum elimina oscilaciones de gradiente en la transición."
    },
    {
        "id": "E5a",
        "name": "Full Pipeline (Continuous)",
        "pipeline_label": "Full Curriculum (P1 + P2 + P3)",
        "phase1_loss": "ce",
        "phase1_epochs": 60,
        "phase1_lr": 1e-3,
        "enable_phase2": True,
        "phase2_epochs": 30,
        "phase2_lr": 2e-4,
        "phase2_optimizer": "continuous",
        "skip_phase3": False,
        "desc": "Pipeline de tres etapas completo con continuidad estricta de optimizador."
    },
    {
        "id": "E5b",
        "name": "Full Pipeline (Reset Opt)",
        "pipeline_label": "Full Curriculum (P1 + P2 + P3)",
        "phase1_loss": "ce",
        "phase1_epochs": 60,
        "phase1_lr": 1e-3,
        "enable_phase2": True,
        "phase2_epochs": 30,
        "phase2_lr": 2e-4,
        "phase2_optimizer": "reset",
        "skip_phase3": False,
        "desc": "Pipeline de tres etapas completo con re-inicialización de optimizador."
    },
    {
        "id": "E6",
        "name": "Direct Pipeline",
        "pipeline_label": "Direct Pipeline (P1[FB] + P3)",
        "phase1_loss": "focal_brier",
        "phase1_epochs": 60,
        "phase1_lr": 1e-3,
        "enable_phase2": False,
        "phase2_epochs": 0,
        "phase2_lr": 0.0,
        "phase2_optimizer": "—",
        "skip_phase3": False,
        "desc": "Evalúa el pipeline de producción actual de OpenJEV (entrenamiento directo + post-hoc)."
    }
]

def format_markdown_table(results: List[Dict[str, Any]]) -> str:
    lines = [
        "### 4.9 Ablation Study: Training Methodology & Decoupled Calibration Matrix\n",
        "| Experiment ID | Pipeline Configuration | Fase 2 Opt | Bool Acc (%) | Choice Acc (%) | Score Acc (%) | ECE (%) [Equal Width] | ECE (%) [Adaptive] | Brier Score | Score Severe Misses |",
        "|---|---|---|---|---|---|---|---|---|---|"
    ]
    for r in results:
        spec = r["spec"]
        m = r["metrics"]
        opt_label = spec["phase2_optimizer"].capitalize() if spec["enable_phase2"] else "—"
        row = (
            f"| **{spec['id']}** | {spec['pipeline_label']} | {opt_label} | "
            f"{m['bool']['acc']:.2f} | {m['choice']['acc']:.2f} | {m['score']['acc']:.2f} | "
            f"{m['summary']['macro_ece_equal_width']:.2f} | {m['summary']['macro_ece_adaptive']:.2f} | "
            f"{m['summary']['macro_brier']:.4f} | {m['score']['severe_misses_pct']:.2f}% |"
        )
        lines.append(row)
    return "\n".join(lines)

def run_matrix(
    model_dir: str = "./models/qwen2.5-0.5b",
    dataset_dir: str = None,
    output_base: str = "./experiments",
    seed: int = 42,
    fast_mode: bool = False,
    selected_experiments: List[str] = None
):
    start_total = time.perf_counter()
    os.makedirs(output_base, exist_ok=True)
    runs_dir = os.path.join(output_base, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    print("================================================================================")
    print("        OpenJEV Scientific Ablation Suite: Decoupled Training Matrix           ")
    print(f"  Model:   {model_dir}")
    print(f"  Seed:    {seed}")
    print(f"  Fast:    {fast_mode}")
    print("================================================================================\n")

    # 1. Load backbone and tokenizer once
    print(f"[Ablation Suite] Loading transformer backbone from '{model_dir}'...")
    t_load = time.perf_counter()
    model, tokenizer = mlx_lm.load(model_dir)
    print(f"[Ablation Suite] Backbone loaded in {time.perf_counter() - t_load:.2f}s.")

    # 2. Extract features once and cache in memory
    print("[Ablation Suite] Extracting representations from frozen backbone (In-Memory Caching)...")
    t_feat = time.perf_counter()
    feature_extractor = HeadTrainer(
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        seed=seed,
        model_instance=model,
        tokenizer_instance=tokenizer
    )
    cached_features = {
        "train": feature_extractor.extract_features("train"),
        "val": feature_extractor.extract_features("val"),
        "test": feature_extractor.extract_features("test")
    }
    print(f"[Ablation Suite] All splits cached in memory in {time.perf_counter() - t_feat:.2f}s.\n")

    # 3. Execute experiments
    results = []
    for spec in EXPERIMENT_SPECS:
        exp_id = spec["id"]
        if selected_experiments and exp_id not in selected_experiments:
            continue

        exp_output_dir = os.path.join(runs_dir, exp_id)
        os.makedirs(exp_output_dir, exist_ok=True)

        p1_epochs = 4 if fast_mode else spec["phase1_epochs"]
        p2_epochs = 2 if fast_mode else spec["phase2_epochs"]
        opt_mode = spec["phase2_optimizer"] if spec["enable_phase2"] else "reset"

        print(f"--------------------------------------------------------------------------------")
        print(f"▶ Running Experiment [{exp_id}]: {spec['name']}")
        print(f"  Configuration: {spec['pipeline_label']} | P2 Opt: {spec['phase2_optimizer']}")
        print(f"  Output Dir:    {exp_output_dir}")
        print(f"--------------------------------------------------------------------------------")

        t_exp = time.perf_counter()
        trainer = HeadTrainer(
            model_dir=model_dir,
            dataset_dir=dataset_dir,
            output_dir=exp_output_dir,
            seed=seed,
            phase1_loss=spec["phase1_loss"],
            phase1_epochs=p1_epochs,
            phase1_lr=spec["phase1_lr"],
            enable_phase2=spec["enable_phase2"],
            phase2_epochs=p2_epochs,
            phase2_lr=spec["phase2_lr"],
            phase2_optimizer=opt_mode,
            skip_phase3=spec["skip_phase3"],
            log_convergence=True,
            model_instance=model,
            tokenizer_instance=tokenizer
        )

        metrics = trainer.train(features=cached_features)
        exp_time = time.perf_counter() - t_exp

        results.append({
            "spec": spec,
            "metrics": metrics,
            "time_seconds": round(exp_time, 2),
            "output_dir": exp_output_dir
        })

    # 4. Generate Markdown table and summary JSON
    markdown_table = format_markdown_table(results)
    summary_path = os.path.join(output_base, "ablation_summary.json")
    table_path = os.path.join(output_base, "ablation_table.md")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(table_path, "w", encoding="utf-8") as f:
        f.write(markdown_table + "\n")

    total_time = time.perf_counter() - start_total
    print("\n================================================================================")
    print(f"  ✅ Ablation Matrix Execution Complete ({len(results)} experiments in {total_time:.2f}s)")
    print(f"  Summary JSON: {summary_path}")
    print(f"  Table MD:     {table_path}")
    print("================================================================================\n")
    print(markdown_table)
    print("\n")
    return results, markdown_table

def main():
    parser = argparse.ArgumentParser(description="OpenJEV Automated Scientific Ablation Matrix Runner")
    parser.add_argument("--model-dir", default="./models/qwen2.5-0.5b", help="Path to backbone model")
    parser.add_argument("--dataset-dir", default=None, help="Dataset directory")
    parser.add_argument("--output-base", default="./experiments", help="Base directory for experiment runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for all experiments")
    parser.add_argument("--fast", action="store_true", help="Fast mode for smoke tests (reduced epochs)")
    parser.add_argument("--experiments", nargs="+", default=None, help="Specific experiment IDs to run (e.g. E1 E4a E4b)")
    args = parser.parse_args()

    run_matrix(
        model_dir=args.model_dir,
        dataset_dir=args.dataset_dir,
        output_base=args.output_base,
        seed=args.seed,
        fast_mode=args.fast,
        selected_experiments=args.experiments
    )

if __name__ == "__main__":
    main()
