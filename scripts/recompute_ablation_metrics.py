#!/usr/bin/env python3
"""
Recompute Ablation Metrics Script
Iterates over all existing experiment checkpoints (E1..E6) across the three models
(Qwen2.5-0.5B, OpenELM-450M, Gemma-2-2B), evaluates them against test split features,
and populates the new Decision & Calibration metrics:
- Bool: F1-score (primary), Accuracy (secondary), Macro F1
- Choice: Top-1 Accuracy (primary)
- Score: Continuous MAE, Severe Misses (>1.5)
- Calibration: ECE (Equal Width, Adaptive), Brier Score

Updates metrics.json, ablation_summary.json, and ablation_table.md for each model,
and triggers aggregation into cross_model_comparison_table.md.
"""

import os
import sys
import json
import time
import pickle
from typing import Dict, List, Any
from safetensors.numpy import load_file
import mlx.core as mx

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_heads import HeadTrainer
from scripts.run_ablation_matrix import format_markdown_table
import scripts.aggregate_ablation_results as aggregator


def recompute_for_model(model_name: str, base_dir: str = "."):
    model_dir = os.path.join(base_dir, "models", model_name)
    dataset_dir = os.path.join(base_dir, "train-dataset2")
    runs_dir = os.path.join(base_dir, "experiments", "run", f"TDS2_{model_name}")
    cache_dir = os.path.join(base_dir, "experiments", "cache")
    cache_file = os.path.join(cache_dir, f"{model_name}_train-dataset2_features.pkl")

    if not os.path.isdir(runs_dir):
        print(f"[Recompute] Skipping {model_name}: {runs_dir} does not exist.")
        return

    print(f"\n{'='*70}")
    print(f"  Recomputing metrics for {model_name}...")
    print(f"  Runs Dir:   {runs_dir}")
    print(f"  Cache File: {cache_file}")
    print(f"{'='*70}")

    # 1. Obtain test features
    features = None
    if os.path.isfile(cache_file):
        print(f"[Recompute] Loading features from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            features = pickle.load(f)
    else:
        print(f"[Recompute] Cache not found. Extracting features for {model_name}...")
        import mlx_lm
        loaded = mlx_lm.load(model_dir)
        trainer_extractor = HeadTrainer(
            model_dir=model_dir,
            dataset_dir=dataset_dir,
            seed=42,
            model_instance=loaded[0],
            tokenizer_instance=loaded[1]
        )
        features = {
            "train": trainer_extractor.extract_features("train"),
            "val": trainer_extractor.extract_features("val"),
            "test": trainer_extractor.extract_features("test")
        }
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(features, f)
        print(f"[Recompute] Features extracted and cached to {cache_file}.")

    test_feat = features["test"]

    # 2. Load existing summary if available to preserve specs and times
    summary_path = os.path.join(runs_dir, "ablation_summary.json")
    existing_by_id: Dict[str, Dict[str, Any]] = {}
    if os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                existing_by_id[item["spec"]["id"]] = item

    # 3. Instantiate trainer for evaluation (no need to reload backbone)
    trainer = HeadTrainer(
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        seed=42
    )

    exp_ids = ["E1", "E2", "E3", "E4a", "E4b", "E5a", "E5b", "E6"]
    results = []

    for eid in exp_ids:
        exp_dir = os.path.join(runs_dir, eid)
        if not os.path.isdir(exp_dir):
            print(f"[Recompute] Warning: {exp_dir} not found, skipping.")
            continue

        heads_path = os.path.join(exp_dir, "heads.safetensors")
        calib_path = os.path.join(exp_dir, "calibration.json")

        if not os.path.isfile(heads_path) or not os.path.isfile(calib_path):
            print(f"[Recompute] Missing weights or calibration for {eid}, skipping.")
            continue

        # Load weights into trainer heads
        tensors = load_file(heads_path)
        trainer.bool_head.linear.weight = mx.array(tensors["bool_head.linear.weight"])
        trainer.bool_head.linear.bias = mx.array(tensors["bool_head.linear.bias"])
        trainer.choice_head.w_choice.weight = mx.array(tensors["choice_head.w_choice.weight"])
        trainer.score_head.w_score.weight = mx.array(tensors["score_head.w_score.weight"])
        trainer.score_head.w_score.bias = mx.array(tensors["score_head.w_score.bias"])

        # Load calibration data
        with open(calib_path, "r", encoding="utf-8") as f:
            calib = json.load(f)
        trainer.calib_data = calib

        # Evaluate on test split
        metrics = trainer.evaluate(test_feat)

        # Save updated metrics.json
        metrics_file = os.path.join(exp_dir, "metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        old_item = existing_by_id.get(eid, {})
        spec = old_item.get("spec", {"id": eid, "pipeline_label": eid, "enable_phase2": False, "phase2_optimizer": "—"})
        time_seconds = old_item.get("time_seconds", 0.0)

        results.append({
            "spec": spec,
            "metrics": metrics,
            "time_seconds": time_seconds,
            "output_dir": exp_dir
        })

        b_f1 = metrics["bool"]["f1"]
        b_acc = metrics["bool"]["acc"]
        c_top1 = metrics["choice"]["top1_acc"]
        s_mae = metrics["score"]["mae_cont"]
        ece = metrics["summary"]["macro_ece_equal_width"]
        print(f"  [{model_name}] {eid:4s} -> Bool F1: {b_f1:.2f}% | Bool Acc: {b_acc:.2f}% | Choice Top-1: {c_top1:.2f}% | Score MAE: {s_mae:.3f} | ECE: {ece:.2f}%")

    # 4. Generate updated Markdown table and summary JSON
    markdown_table = format_markdown_table(results, model_name=model_name)

    model_summary_file = f"ablation_summary_{model_name}.json"
    model_table_file = f"ablation_table_{model_name}.md"

    summary_paths = [
        os.path.join(base_dir, "experiments", model_summary_file),
        os.path.join(runs_dir, "ablation_summary.json")
    ]
    table_paths = [
        os.path.join(base_dir, "experiments", model_table_file),
        os.path.join(runs_dir, "ablation_table.md")
    ]

    for p in summary_paths:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    for p in table_paths:
        with open(p, "w", encoding="utf-8") as f:
            f.write(markdown_table + "\n")

    # Auto-promote E5b
    e5b_dir = os.path.join(runs_dir, "E5b")
    if os.path.isdir(e5b_dir) and os.path.isdir(model_dir):
        import shutil
        for art in ["heads.safetensors", "calibration.json", "heads_metadata.json"]:
            src = os.path.join(e5b_dir, art)
            dst = os.path.join(model_dir, art)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        print(f"  [Auto-Promote] Synced E5b production artifacts to {model_dir}")

    print(f"\nUpdated {len(results)} experiments for {model_name}.")
    print(markdown_table)


def main():
    models = ["qwen2.5-0.5b", "openelm-450m", "gemma-2-2b"]
    for m in models:
        recompute_for_model(m)

    # Re-run aggregator
    print(f"\n{'='*70}")
    print("  Aggregating cross-model ablation results...")
    print(f"{'='*70}\n")
    aggregator.main()


if __name__ == "__main__":
    main()
