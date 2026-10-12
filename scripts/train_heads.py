"""
OpenJEV Supervised Head Training & Calibration Script (Python / MLX)
Trains the canonical decision heads on Apple Silicon Metal GPU with frozen backbone:
1. BoolHead (Noul): Trained with Combined Focal + Brier Loss
2. ChoiceHead: Trained with Cross-Entropy over L2-normalized bilinear projections
3. ScoreHead: Trained with Ordinal distance-penalized score loss

Followed by automatic Post-Hoc Temperature Calibration (SPEC.md §8 & ARCHITECTURE.md §4).
"""

import os
import json
import time
import argparse
from typing import Dict, List, Any, Tuple
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from safetensors.numpy import save_file

from core.heads import BoolHead, ChoiceHead, ScoreHead
from core.losses import choice_cross_entropy_loss, ordinal_score_loss
from calibration.loss_shaping import combined_focal_brier_loss
from calibration.post_hoc import TemperatureScaler
from core.model_scanner import get_model_hidden_size, get_expected_shapes

from core.batching.packer import extract_question_text

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

class HeadTrainer:
    def __init__(self, model_dir: str, dataset_dir: str, lr: float = 1e-3, epochs: int = 100):
        self.model_dir = model_dir
        self.dataset_dir = dataset_dir
        self.lr = lr
        self.epochs = epochs

        self.hidden_dim = get_model_hidden_size(model_dir, default=896)
        print(f"[OpenJEV Trainer] Initializing training for model in '{model_dir}' (hidden_dim={self.hidden_dim})")

        # Load Backbone & Tokenizer
        import mlx_lm
        self.model, self.tokenizer = mlx_lm.load(model_dir)
        self.backbone = self.model.model
        backbone_name = self.backbone.__class__.__name__
        print(f"[OpenJEV Trainer] Loaded {backbone_name} transformer backbone on Metal GPU (Frozen)")

        # Instantiate Decision Heads
        self.bool_head = BoolHead(hidden_dim=self.hidden_dim)
        self.choice_head = ChoiceHead(hidden_dim=self.hidden_dim)
        self.score_head = ScoreHead(hidden_dim=self.hidden_dim, num_levels=5)

    def _extract_embedding(self, state: str, q_text: str = "") -> mx.array:
        if q_text:
            s_tokens = self.tokenizer.encode(state, add_special_tokens=False)
            q_tokens = self.tokenizer.encode(f"\n\nQuestion: {q_text}", add_special_tokens=False)
            tokens = (s_tokens + q_tokens) or [1]
        else:
            tokens = self.tokenizer.encode(state, add_special_tokens=False) or [1]
        hidden = self.backbone(mx.array([tokens]))
        return hidden[:, -1, :].astype(mx.float32)

    def extract_features(self, split_name: str) -> Dict[str, List[Any]]:
        path = os.path.join(self.dataset_dir, f"{split_name}.jsonl")
        records = load_jsonl(path)
        print(f"[OpenJEV Trainer] Extracting features for {len(records)} records in '{split_name}'...")

        features: Dict[str, List[Any]] = {
            "noul": [],
            "choice": [],
            "score": []
        }

        # Cache option embeddings to avoid redundant passes
        option_cache: Dict[str, mx.array] = {}

        t0 = time.perf_counter()
        for idx, rec in enumerate(records):
            prim = rec["primitive"]
            state = rec["state"]
            q = rec["question"]
            q_text = extract_question_text(q)
            h_q = self._extract_embedding(state, q_text)

            if prim == "noul":
                target_val = float(rec["target"])
                features["noul"].append((h_q, target_val))
            elif prim == "choice":
                criteria = q.get("criteria", {})
                options = list(criteria.items())
                target_id = rec["target"]
                target_idx = next(i for i, (oid, _) in enumerate(options) if oid == target_id)

                opt_vecs = []
                for oid, desc in options:
                    opt_str = f"{oid}: {desc}"
                    if opt_str not in option_cache:
                        option_cache[opt_str] = self._extract_embedding(opt_str)
                    opt_vecs.append(option_cache[opt_str])
                e_options = mx.concatenate(opt_vecs, axis=0) # (N, d)
                features["choice"].append((h_q, e_options, target_idx))
            elif prim == "score":
                criteria = q.get("criteria", [])
                target_level = int(rec["target"])
                num_levels = len(criteria)
                features["score"].append((h_q, num_levels, target_level))

            if (idx + 1) % 200 == 0 or (idx + 1) == len(records):
                print(f"  Processed {idx + 1}/{len(records)} samples ({time.perf_counter() - t0:.1f}s)")

        return features

    def train(self):
        train_feat = self.extract_features("train")
        val_feat = self.extract_features("val")

        print("\n[OpenJEV Trainer] Starting Supervised Training on Metal GPU...")
        start_time = time.perf_counter()

        # 1. Train BoolHead (Noul)
        opt_bool = optim.AdamW(learning_rate=self.lr)
        noul_train = train_feat["noul"]
        h_q_noul = mx.concatenate([item[0] for item in noul_train], axis=0)
        t_noul = mx.array([item[1] for item in noul_train])

        def bool_loss_fn(model, x, y):
            logits = model.linear(x).squeeze(-1)
            p_true = mx.sigmoid(logits)
            return combined_focal_brier_loss(p_true, y)

        bool_grad_fn = nn.value_and_grad(self.bool_head, bool_loss_fn)

        for ep in range(1, self.epochs + 1):
            loss, grads = bool_grad_fn(self.bool_head, h_q_noul, t_noul)
            opt_bool.update(self.bool_head, grads)
            mx.eval(self.bool_head.parameters(), opt_bool.state)
        print(f"  ✅ NoulHead trained: final loss = {loss.item():.4f}")

        # 2. Train ChoiceHead (Batched by option count)
        opt_choice = optim.AdamW(learning_rate=self.lr)
        choice_train = train_feat["choice"]

        choice_by_n: Dict[int, Dict[str, List[Any]]] = {}
        for h_q, e_options, target_idx in choice_train:
            n_opts = e_options.shape[0]
            if n_opts not in choice_by_n:
                choice_by_n[n_opts] = {"h_q": [], "e_options": [], "targets": []}
            choice_by_n[n_opts]["h_q"].append(h_q)
            choice_by_n[n_opts]["e_options"].append(mx.expand_dims(e_options, 0))
            choice_by_n[n_opts]["targets"].append(target_idx)

        batched_choice_train = [
            (
                mx.concatenate(data["h_q"], axis=0),
                mx.concatenate(data["e_options"], axis=0),
                mx.array(data["targets"])
            )
            for n_opts, data in choice_by_n.items()
        ]

        def choice_step_fn(model, batches):
            total_loss = 0.0
            total_samples = 0
            for h_q_b, e_opts_b, targets_b in batches:
                sel, conf, probs = model(h_q_b, e_opts_b)
                loss = choice_cross_entropy_loss(probs, targets_b)
                total_loss = total_loss + loss * targets_b.shape[0]
                total_samples += targets_b.shape[0]
            return total_loss / max(total_samples, 1)

        choice_grad_fn = nn.value_and_grad(self.choice_head, choice_step_fn)

        for ep in range(1, self.epochs + 1):
            loss_c, grads_c = choice_grad_fn(self.choice_head, batched_choice_train)
            opt_choice.update(self.choice_head, grads_c)
            mx.eval(self.choice_head.parameters(), opt_choice.state)
        print(f"  ✅ ChoiceHead trained: final loss = {loss_c.item():.4f}")

        # 3. Train ScoreHead (Batched by rubric levels)
        opt_score = optim.AdamW(learning_rate=self.lr)
        score_train = train_feat["score"]

        score_by_levels: Dict[int, Dict[str, List[Any]]] = {}
        for h_q, num_levels, target_level in score_train:
            if num_levels not in score_by_levels:
                score_by_levels[num_levels] = {"h_q": [], "targets": []}
            score_by_levels[num_levels]["h_q"].append(h_q)
            score_by_levels[num_levels]["targets"].append(target_level)

        batched_score_train = [
            (
                mx.concatenate(data["h_q"], axis=0),
                num_levels,
                mx.array(data["targets"])
            )
            for num_levels, data in score_by_levels.items()
        ]

        def score_step_fn(model, batches):
            total_loss = 0.0
            total_samples = 0
            for h_q_b, num_levels, targets_b in batches:
                s_out = model(h_q_b, min_val=0, max_val=num_levels - 1, num_levels=num_levels)
                loss = ordinal_score_loss(s_out["logits"], targets_b, num_levels=num_levels, from_logits=True)
                total_loss = total_loss + loss * targets_b.shape[0]
                total_samples += targets_b.shape[0]
            return total_loss / max(total_samples, 1)

        score_grad_fn = nn.value_and_grad(self.score_head, score_step_fn)

        for ep in range(1, self.epochs + 1):
            loss_s, grads_s = score_grad_fn(self.score_head, batched_score_train)
            opt_score.update(self.score_head, grads_s)
            mx.eval(self.score_head.parameters(), opt_score.state)
        print(f"  ✅ ScoreHead trained: final loss = {loss_s.item():.4f}")

        # Validation Accuracy Check
        val_acc_noul = self._eval_noul_acc(val_feat["noul"])
        val_acc_choice = self._eval_choice_acc(val_feat["choice"])
        val_acc_score = self._eval_score_acc(val_feat["score"])

        print(f"\n[OpenJEV Trainer] Validation Accuracy:")
        print(f"  - Noul:   {val_acc_noul * 100:.2f}%")
        print(f"  - Choice: {val_acc_choice * 100:.2f}%")
        print(f"  - Score:  {val_acc_score * 100:.2f}%")

        # 4. Post-Hoc Temperature Calibration
        print("\n[OpenJEV Trainer] Running Post-Hoc Temperature Calibration...")
        calib_data = self._calibrate(val_feat)

        # 5. Export Trained Heads & Calibration
        self._export(calib_data)
        elapsed = time.perf_counter() - start_time
        print(f"\n🎉 [OpenJEV Trainer] Training and calibration completed in {elapsed:.2f}s!")

    def _eval_noul_acc(self, val_samples: List[Any]) -> float:
        correct = 0
        for h_q, target in val_samples:
            vals, confs, p_true = self.bool_head(h_q)
            pred = 1 if float(p_true[0].item()) >= 0.5 else 0
            if pred == int(target):
                correct += 1
        return correct / len(val_samples) if val_samples else 1.0

    def _eval_choice_acc(self, val_samples: List[Any]) -> float:
        correct = 0
        for h_q, e_options, target_idx in val_samples:
            sel, conf, probs = self.choice_head(h_q, e_options)
            if int(sel[0].item()) == target_idx:
                correct += 1
        return correct / len(val_samples) if val_samples else 1.0

    def _eval_score_acc(self, val_samples: List[Any]) -> float:
        correct = 0
        for h_q, num_levels, target_level in val_samples:
            s_out = self.score_head(h_q, min_val=0, max_val=num_levels - 1, num_levels=num_levels)
            pred_level = int(s_out["discrete_level"][0].item())
            if pred_level == target_level:
                correct += 1
        return correct / len(val_samples) if val_samples else 1.0

    def _calibrate(self, val_feat: Dict[str, List[Any]]) -> Dict[str, Any]:
        # Fit Noul temperature
        noul_logits = []
        noul_targets = []
        for h_q, target in val_feat["noul"]:
            logit = float(self.bool_head.linear(h_q).squeeze(-1)[0].item())
            noul_logits.append(logit)
            noul_targets.append(target)
        scaler_bool = TemperatureScaler()
        t_bool = scaler_bool.fit(mx.array(noul_logits), mx.array(noul_targets))

        calib_dict = {
            "version": "1.0",
            "model": os.path.basename(self.model_dir),
            "status": "calibrated",
            "temperatures": {
                "bool": round(t_bool, 4),
                "choice": 1.0,
                "score": 1.0
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        calib_path = os.path.join(self.model_dir, "calibration.json")
        with open(calib_path, "w", encoding="utf-8") as f:
            json.dump(calib_dict, f, indent=2)
        print(f"  Saved calibration to {calib_path} (T_bool={t_bool:.4f})")
        return calib_dict

    def _export(self, calib_data: Dict[str, Any]):
        heads_path = os.path.join(self.model_dir, "heads.safetensors")
        tensors = {
            "bool_head.linear.weight": np.array(self.bool_head.linear.weight),
            "bool_head.linear.bias": np.array(self.bool_head.linear.bias),
            "choice_head.w_choice.weight": np.array(self.choice_head.w_choice.weight),
            "score_head.w_score.weight": np.array(self.score_head.w_score.weight),
            "score_head.w_score.bias": np.array(self.score_head.w_score.bias)
        }
        save_file(tensors, heads_path)
        print(f"  Saved trained weights to {heads_path}")

        meta_path = os.path.join(self.model_dir, "heads_metadata.json")
        meta = {
            "format_version": "1.0",
            "openjev_version": "2.0.0",
            "hidden_dim": self.hidden_dim,
            "status": "trained_and_calibrated",
            "tensors": {k: list(v.shape) for k, v in tensors.items()},
            "calibration": calib_data
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="OpenJEV Supervised Head Training & Calibration CLI")
    parser.add_argument("--model-dir", default="./models/qwen2.5-0.5b", help="Path to model directory")
    parser.add_argument("--dataset-dir", default="./train-dataset", help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    trainer = HeadTrainer(
        model_dir=args.model_dir,
        dataset_dir=args.dataset_dir,
        lr=args.lr,
        epochs=args.epochs
    )
    trainer.train()

if __name__ == "__main__":
    main()
