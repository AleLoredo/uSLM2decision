"""
OpenJEV Supervised Head Training & Calibration Pipeline (Python / MLX)
Decoupled Modular Architecture supporting:
- Phase 1: Linear Probing with Loss Selector (CE vs Focal+Brier)
- Phase 2: Warm-Start Curriculum Refinement (continuous vs reset optimizer state)
- Phase 3: Post-Hoc Temperature Calibration
Per SPEC.md §8, ARCHITECTURE.md §3-§5, and decoupled_training_and_ablation_plan.md.
"""

import os
import sys
import csv
import json
import time
import argparse
from typing import Dict, List, Any, Tuple, Optional

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as utils
from safetensors.numpy import save_file

from core.heads import BoolHead, ChoiceHead, ScoreHead
from core.losses import (
    binary_cross_entropy_loss,
    bool_loss,
    choice_loss,
    choice_cross_entropy_loss,
    ordinal_score_loss,
)
from calibration.loss_shaping import combined_focal_brier_loss
from calibration.post_hoc import TemperatureScaler
from calibration.metrics import compute_ece, compute_brier_decomposition
from core.model_scanner import get_model_hidden_size
from core.batching.packer import extract_question_text

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

class HeadTrainer:
    def __init__(
        self,
        model_dir: str = "./models/qwen2.5-0.5b",
        dataset_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        seed: int = 42,
        phase1_loss: str = "focal_brier",
        phase1_epochs: int = 60,
        phase1_lr: float = 1e-3,
        enable_phase2: bool = False,
        phase2_epochs: int = 30,
        phase2_lr: float = 2e-4,
        phase2_optimizer: str = "reset",
        skip_phase3: bool = False,
        fit_score_temperature: bool = False,
        log_convergence: bool = False,
        model_instance: Optional[Any] = None,
        tokenizer_instance: Optional[Any] = None,
    ):
        self.model_dir = model_dir
        self.output_dir = output_dir or model_dir
        self.seed = seed
        self.phase1_loss = phase1_loss.lower()
        self.phase1_epochs = phase1_epochs
        self.phase1_lr = phase1_lr
        self.enable_phase2 = enable_phase2
        self.phase2_epochs = phase2_epochs
        self.phase2_lr = phase2_lr
        self.phase2_optimizer = phase2_optimizer.lower()
        self.skip_phase3 = skip_phase3
        self.fit_score_temperature = fit_score_temperature
        self.log_convergence = log_convergence

        # Resolve dataset directory if not provided
        if dataset_dir is None:
            candidates = ["./train-dataset-bkup", "./train-dataset1", "./train-dataset"]
            for cand in candidates:
                if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "train.jsonl")):
                    dataset_dir = cand
                    break
            if dataset_dir is None:
                dataset_dir = "./train-dataset"
        self.dataset_dir = dataset_dir

        # Fix random seeds for reproducible science
        mx.random.seed(self.seed)
        np.random.seed(self.seed)

        self.hidden_dim = get_model_hidden_size(model_dir, default=896)
        print(f"[OpenJEV Trainer] Model dir: '{model_dir}' (hidden_dim={self.hidden_dim}) | Dataset: '{self.dataset_dir}'")

        # Load or reuse backbone
        if model_instance is not None and tokenizer_instance is not None:
            self.model = model_instance
            self.tokenizer = tokenizer_instance
            self.backbone = getattr(self.model, "model", getattr(self.model, "transformer", None))
        else:
            self.model = None
            self.tokenizer = None
            self.backbone = None

        # Instantiate decision heads
        self.bool_head = BoolHead(hidden_dim=self.hidden_dim)
        self.choice_head = ChoiceHead(hidden_dim=self.hidden_dim)
        self.score_head = ScoreHead(hidden_dim=self.hidden_dim, num_levels=5)

        # Optimizers initialized in phase 1
        self.opt_bool = optim.AdamW(learning_rate=self.phase1_lr)
        self.opt_choice = optim.AdamW(learning_rate=self.phase1_lr)
        self.opt_score = optim.AdamW(learning_rate=self.phase1_lr)

        self.convergence_records: List[Dict[str, Any]] = []
        self.calib_data: Dict[str, Any] = {
            "version": "1.0",
            "model": os.path.basename(self.model_dir),
            "status": "uncalibrated",
            "temperatures": {"bool": 1.0, "choice": 1.0, "score": 1.0},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

    def _ensure_backbone_loaded(self):
        if self.backbone is None:
            import mlx_lm
            print(f"[OpenJEV Trainer] Loading backbone on Metal GPU from '{self.model_dir}'...")
            self.model, self.tokenizer = mlx_lm.load(self.model_dir)
            self.backbone = getattr(self.model, "model", getattr(self.model, "transformer", None))
            if self.backbone is None:
                raise AttributeError(f"Could not extract transformer backbone from {type(self.model)}")
            print(f"[OpenJEV Trainer] Backbone {self.backbone.__class__.__name__} ready.")

    def _extract_embedding(self, state: str, q_text: str = "") -> mx.array:
        self._ensure_backbone_loaded()
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
            "bool": [],
            "choice": [],
            "score": []
        }
        option_cache: Dict[str, mx.array] = {}
        t0 = time.perf_counter()

        for idx, rec in enumerate(records):
            prim = rec["primitive"]
            state = rec["state"]
            q = rec["question"]
            q_text = extract_question_text(q)
            h_q = self._extract_embedding(state, q_text)

            if prim in ("bool", "noul"):
                target_val = float(rec["target"])
                features["bool"].append((h_q, target_val))
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

            if (idx + 1) % 400 == 0 or (idx + 1) == len(records):
                print(f"  Processed {idx + 1}/{len(records)} samples ({time.perf_counter() - t0:.1f}s)")

        return features

    def _calc_grad_norm(self, grads: Any) -> float:
        flat = utils.tree_flatten(grads)
        sq_sum = sum(mx.sum(val ** 2) for _, val in flat if val is not None)
        return float(mx.sqrt(sq_sum).item())

    def _prepare_batches(self, split_data: Dict[str, List[Any]]):
        # 1. Bool batch
        bool_train = split_data.get("bool", split_data.get("noul", []))
        if bool_train:
            h_q_bool = mx.concatenate([item[0] for item in bool_train], axis=0)
            t_bool = mx.array([item[1] for item in bool_train])
        else:
            h_q_bool, t_bool = None, None

        # 2. Choice batches grouped by option count
        choice_train = split_data.get("choice", [])
        choice_by_n: Dict[int, Dict[str, List[Any]]] = {}
        for h_q, e_options, target_idx in choice_train:
            n_opts = e_options.shape[0]
            if n_opts not in choice_by_n:
                choice_by_n[n_opts] = {"h_q": [], "e_options": [], "targets": []}
            choice_by_n[n_opts]["h_q"].append(h_q)
            choice_by_n[n_opts]["e_options"].append(mx.expand_dims(e_options, 0))
            choice_by_n[n_opts]["targets"].append(target_idx)

        batched_choice = [
            (
                mx.concatenate(data["h_q"], axis=0),
                mx.concatenate(data["e_options"], axis=0),
                mx.array(data["targets"])
            )
            for n_opts, data in choice_by_n.items()
        ]

        # 3. Score batches grouped by rubric levels
        score_train = split_data.get("score", [])
        score_by_levels: Dict[int, Dict[str, List[Any]]] = {}
        for h_q, num_levels, target_level in score_train:
            if num_levels not in score_by_levels:
                score_by_levels[num_levels] = {"h_q": [], "targets": []}
            score_by_levels[num_levels]["h_q"].append(h_q)
            score_by_levels[num_levels]["targets"].append(target_level)

        batched_score = [
            (
                mx.concatenate(data["h_q"], axis=0),
                num_levels,
                mx.array(data["targets"])
            )
            for num_levels, data in score_by_levels.items()
        ]

        return (h_q_bool, t_bool), batched_choice, batched_score

    def run_phase1(self, train_feat: Dict[str, List[Any]]):
        print(f"\n[OpenJEV Trainer] === FASE 1: Linear Probing ({self.phase1_loss.upper()}, {self.phase1_epochs} épocas, lr={self.phase1_lr}) ===")
        (h_q_bool, t_bool), batched_choice, batched_score = self._prepare_batches(train_feat)

        # 1. BoolHead
        if h_q_bool is not None:
            def bool_loss_fn(model, x, y):
                logits = model.linear(x).squeeze(-1)
                p_true = mx.sigmoid(logits)
                return bool_loss(p_true, y, loss_type=self.phase1_loss, gamma=1.0, brier_weight=0.5)

            bool_grad_fn = nn.value_and_grad(self.bool_head, bool_loss_fn)
            for ep in range(1, self.phase1_epochs + 1):
                loss_b, grads_b = bool_grad_fn(self.bool_head, h_q_bool, t_bool)
                self.opt_bool.update(self.bool_head, grads_b)
                mx.eval(self.bool_head.parameters(), self.opt_bool.state)
                if self.log_convergence:
                    gnorm = self._calc_grad_norm(grads_b)
                    self.convergence_records.append({
                        "phase": "phase1", "epoch": ep, "primitive": "bool",
                        "loss": float(loss_b.item()), "grad_norm": gnorm, "opt_mode": "p1"
                    })
            print(f"  ✅ BoolHead trained (Phase 1): final loss = {loss_b.item():.4f}")

        # 2. ChoiceHead
        if batched_choice:
            def choice_step_fn(model, batches):
                total_loss = 0.0
                total_samples = 0
                for h_q_b, e_opts_b, targets_b in batches:
                    sel, conf, probs = model(h_q_b, e_opts_b)
                    loss = choice_loss(probs, targets_b, loss_type=self.phase1_loss, brier_weight=0.5)
                    total_loss = total_loss + loss * targets_b.shape[0]
                    total_samples += targets_b.shape[0]
                return total_loss / max(total_samples, 1)

            choice_grad_fn = nn.value_and_grad(self.choice_head, choice_step_fn)
            for ep in range(1, self.phase1_epochs + 1):
                loss_c, grads_c = choice_grad_fn(self.choice_head, batched_choice)
                self.opt_choice.update(self.choice_head, grads_c)
                mx.eval(self.choice_head.parameters(), self.opt_choice.state)
                if self.log_convergence:
                    gnorm = self._calc_grad_norm(grads_c)
                    self.convergence_records.append({
                        "phase": "phase1", "epoch": ep, "primitive": "choice",
                        "loss": float(loss_c.item()), "grad_norm": gnorm, "opt_mode": "p1"
                    })
            print(f"  ✅ ChoiceHead trained (Phase 1): final loss = {loss_c.item():.4f}")

        # 3. ScoreHead
        if batched_score:
            score_mode = "one_hot" if self.phase1_loss == "ce" else "ordinal_gaussian"
            def score_step_fn(model, batches):
                total_loss = 0.0
                total_samples = 0
                for h_q_b, num_levels, targets_b in batches:
                    s_out = model(h_q_b, min_val=0, max_val=num_levels - 1, num_levels=num_levels)
                    loss = ordinal_score_loss(
                        s_out["logits"], targets_b, num_levels=num_levels,
                        from_logits=True, mode=score_mode, sigma=0.35
                    )
                    total_loss = total_loss + loss * targets_b.shape[0]
                    total_samples += targets_b.shape[0]
                return total_loss / max(total_samples, 1)

            score_grad_fn = nn.value_and_grad(self.score_head, score_step_fn)
            for ep in range(1, self.phase1_epochs + 1):
                loss_s, grads_s = score_grad_fn(self.score_head, batched_score)
                self.opt_score.update(self.score_head, grads_s)
                mx.eval(self.score_head.parameters(), self.opt_score.state)
                if self.log_convergence:
                    gnorm = self._calc_grad_norm(grads_s)
                    self.convergence_records.append({
                        "phase": "phase1", "epoch": ep, "primitive": "score",
                        "loss": float(loss_s.item()), "grad_norm": gnorm, "opt_mode": "p1"
                    })
            print(f"  ✅ ScoreHead trained (Phase 1): final loss = {loss_s.item():.4f}")

    def run_phase2(self, train_feat: Dict[str, List[Any]]):
        print(f"\n[OpenJEV Trainer] === FASE 2: Warm-Start Curriculum Refinement ({self.phase2_epochs} épocas, lr={self.phase2_lr}, opt={self.phase2_optimizer}) ===")
        (h_q_bool, t_bool), batched_choice, batched_score = self._prepare_batches(train_feat)

        # Optimizer transition: Reset vs Continuous
        if self.phase2_optimizer == "reset":
            print(f"  [Phase 2] Re-inicializando estado de AdamW con lr={self.phase2_lr} (Modalidad A: Reset)")
            self.opt_bool = optim.AdamW(learning_rate=self.phase2_lr)
            self.opt_choice = optim.AdamW(learning_rate=self.phase2_lr)
            self.opt_score = optim.AdamW(learning_rate=self.phase2_lr)
        elif self.phase2_optimizer == "continuous":
            print(f"  [Phase 2] Preservando buffers de momentum de AdamW con lr={self.phase2_lr} (Modalidad B: Continuous)")
            self.opt_bool.learning_rate = self.phase2_lr
            self.opt_choice.learning_rate = self.phase2_lr
            self.opt_score.learning_rate = self.phase2_lr
        else:
            raise ValueError(f"Unknown phase2_optimizer: {self.phase2_optimizer}")

        # Phase 2 always applies calibrated objective functions (Focal+Brier and Gaussian ordinal smoothing)
        if h_q_bool is not None:
            def bool_loss_fn(model, x, y):
                logits = model.linear(x).squeeze(-1)
                p_true = mx.sigmoid(logits)
                return combined_focal_brier_loss(p_true, y, gamma=1.0, brier_weight=0.5)

            bool_grad_fn = nn.value_and_grad(self.bool_head, bool_loss_fn)
            for ep in range(1, self.phase2_epochs + 1):
                loss_b, grads_b = bool_grad_fn(self.bool_head, h_q_bool, t_bool)
                self.opt_bool.update(self.bool_head, grads_b)
                mx.eval(self.bool_head.parameters(), self.opt_bool.state)
                if self.log_convergence:
                    gnorm = self._calc_grad_norm(grads_b)
                    self.convergence_records.append({
                        "phase": "phase2", "epoch": ep, "primitive": "bool",
                        "loss": float(loss_b.item()), "grad_norm": gnorm, "opt_mode": self.phase2_optimizer
                    })
            print(f"  ✅ BoolHead refined (Phase 2): final loss = {loss_b.item():.4f}")

        if batched_choice:
            def choice_step_fn(model, batches):
                total_loss = 0.0
                total_samples = 0
                for h_q_b, e_opts_b, targets_b in batches:
                    sel, conf, probs = model(h_q_b, e_opts_b)
                    loss = choice_loss(probs, targets_b, loss_type="focal_brier", brier_weight=0.5)
                    total_loss = total_loss + loss * targets_b.shape[0]
                    total_samples += targets_b.shape[0]
                return total_loss / max(total_samples, 1)

            choice_grad_fn = nn.value_and_grad(self.choice_head, choice_step_fn)
            for ep in range(1, self.phase2_epochs + 1):
                loss_c, grads_c = choice_grad_fn(self.choice_head, batched_choice)
                self.opt_choice.update(self.choice_head, grads_c)
                mx.eval(self.choice_head.parameters(), self.opt_choice.state)
                if self.log_convergence:
                    gnorm = self._calc_grad_norm(grads_c)
                    self.convergence_records.append({
                        "phase": "phase2", "epoch": ep, "primitive": "choice",
                        "loss": float(loss_c.item()), "grad_norm": gnorm, "opt_mode": self.phase2_optimizer
                    })
            print(f"  ✅ ChoiceHead refined (Phase 2): final loss = {loss_c.item():.4f}")

        if batched_score:
            def score_step_fn(model, batches):
                total_loss = 0.0
                total_samples = 0
                for h_q_b, num_levels, targets_b in batches:
                    s_out = model(h_q_b, min_val=0, max_val=num_levels - 1, num_levels=num_levels)
                    loss = ordinal_score_loss(
                        s_out["logits"], targets_b, num_levels=num_levels,
                        from_logits=True, mode="ordinal_gaussian", sigma=0.35
                    )
                    total_loss = total_loss + loss * targets_b.shape[0]
                    total_samples += targets_b.shape[0]
                return total_loss / max(total_samples, 1)

            score_grad_fn = nn.value_and_grad(self.score_head, score_step_fn)
            for ep in range(1, self.phase2_epochs + 1):
                loss_s, grads_s = score_grad_fn(self.score_head, batched_score)
                self.opt_score.update(self.score_head, grads_s)
                mx.eval(self.score_head.parameters(), self.opt_score.state)
                if self.log_convergence:
                    gnorm = self._calc_grad_norm(grads_s)
                    self.convergence_records.append({
                        "phase": "phase2", "epoch": ep, "primitive": "score",
                        "loss": float(loss_s.item()), "grad_norm": gnorm, "opt_mode": self.phase2_optimizer
                    })
            print(f"  ✅ ScoreHead refined (Phase 2): final loss = {loss_s.item():.4f}")

    def run_phase3(self, val_feat: Dict[str, List[Any]]) -> Dict[str, Any]:
        if self.skip_phase3:
            print("\n[OpenJEV Trainer] === FASE 3: Calibración Omitida (--skip-phase3, T=1.0) ===")
            self.calib_data = {
                "version": "1.0",
                "model": os.path.basename(self.model_dir),
                "status": "uncalibrated_post_hoc",
                "temperatures": {"bool": 1.0, "choice": 1.0, "score": 1.0},
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            return self.calib_data

        print("\n[OpenJEV Trainer] === FASE 3: Calibración Extrínseca Post-Hoc (Grid Search en ECE) ===")
        # 1. Fit Bool temperature
        bool_samples = val_feat.get("bool", val_feat.get("noul", []))
        noul_logits, noul_targets = [], []
        for h_q, target in bool_samples:
            logit = float(self.bool_head.linear(h_q).squeeze(-1)[0].item())
            noul_logits.append(logit)
            noul_targets.append(float(target))

        scaler_bool = TemperatureScaler()
        t_bool = scaler_bool.fit(mx.array(noul_logits), mx.array(noul_targets))
        print(f"  Fitted T_bool = {t_bool:.4f}")

        t_choice = 1.0
        t_score = 1.0

        self.calib_data = {
            "version": "1.0",
            "model": os.path.basename(self.model_dir),
            "status": "calibrated",
            "temperatures": {
                "bool": round(float(t_bool), 4),
                "choice": round(float(t_choice), 4),
                "score": round(float(t_score), 4)
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        return self.calib_data

    def evaluate(self, eval_feat: Dict[str, List[Any]]) -> Dict[str, Any]:
        t_bool = self.calib_data["temperatures"].get("bool", 1.0)
        t_choice = self.calib_data["temperatures"].get("choice", 1.0)
        t_score = self.calib_data["temperatures"].get("score", 1.0)

        # 1. Evaluate Bool
        bool_samples = eval_feat.get("bool", eval_feat.get("noul", []))
        bool_preds, bool_probs, bool_targets = [], [], []
        for h_q, target in bool_samples:
            logit = float(self.bool_head.linear(h_q).squeeze(-1)[0].item())
            p_calib = 1.0 / (1.0 + np.exp(-logit / t_bool))
            pred = 1 if p_calib >= 0.5 else 0
            bool_preds.append(pred)
            bool_probs.append(p_calib)
            bool_targets.append(float(target))

        bool_acc = np.mean(np.array(bool_preds) == np.array(bool_targets)) if bool_samples else 1.0
        bool_ece_ew = compute_ece(np.array(bool_probs), np.array(bool_targets), scheme="equal_width")["ece"]
        bool_ece_ad = compute_ece(np.array(bool_probs), np.array(bool_targets), scheme="adaptive")["ece"]
        bool_brier = compute_brier_decomposition(np.array(bool_probs), np.array(bool_targets))["brier_score"]

        # 2. Evaluate Choice
        choice_samples = eval_feat.get("choice", [])
        choice_preds, choice_targets = [], []
        choice_probs_list = []
        for h_q, e_options, target_idx in choice_samples:
            sel, conf, probs = self.choice_head(h_q, e_options, temperature=t_choice)
            pred = int(sel[0].item())
            choice_preds.append(pred)
            choice_targets.append(int(target_idx))
            choice_probs_list.append(np.array(probs[0]))

        choice_acc = np.mean(np.array(choice_preds) == np.array(choice_targets)) if choice_samples else 1.0
        choice_confs = np.array([p[t] if t < len(p) else 0.0 for p, t in zip(choice_probs_list, choice_preds)])
        choice_correct = (np.array(choice_preds) == np.array(choice_targets)).astype(float)
        choice_ece_ew = compute_ece(choice_confs, choice_correct, scheme="equal_width")["ece"] if len(choice_confs) > 0 else 0.0
        choice_ece_ad = compute_ece(choice_confs, choice_correct, scheme="adaptive")["ece"] if len(choice_confs) > 0 else 0.0
        choice_brier = np.mean([
            np.sum((p - np.eye(len(p))[t]) ** 2) for p, t in zip(choice_probs_list, choice_targets)
        ]) if choice_samples else 0.0

        # 3. Evaluate Score (Continuous Expected Value per SPEC.md §3.2 + Discrete Argmax)
        score_samples = eval_feat.get("score", [])
        score_preds, score_targets = [], []
        score_diffs_disc, score_diffs_cont = [], []
        for h_q, num_levels, target_level in score_samples:
            s_out = self.score_head(h_q, min_val=0, max_val=num_levels - 1, num_levels=num_levels)
            logits = np.array(s_out["logits"][0])
            scaled_logits = logits / t_score
            probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))
            
            # Discrete argmax level
            pred_level = int(np.argmax(probs))
            score_preds.append(pred_level)
            score_targets.append(int(target_level))
            score_diffs_disc.append(abs(pred_level - int(target_level)))

            # Continuous expected value: S_bar = sum(k * p_k)
            levels = np.arange(len(probs))
            s_bar = float(np.sum(probs * levels))
            score_diffs_cont.append(abs(s_bar - float(target_level)))

        diffs_disc = np.array(score_diffs_disc) if score_diffs_disc else np.array([0])
        diffs_cont = np.array(score_diffs_cont) if score_diffs_cont else np.array([0.0])

        score_acc = np.mean(diffs_disc == 0) if score_samples else 1.0
        off1_misses = np.mean(diffs_disc == 1) * 100.0 if score_samples else 0.0
        mae_disc = float(np.mean(diffs_disc)) if score_samples else 0.0

        mae_cont = float(np.mean(diffs_cont)) if score_samples else 0.0
        rmse_cont = float(np.sqrt(np.mean(diffs_cont ** 2))) if score_samples else 0.0
        tolerance_05_pct = float(np.mean(diffs_cont > 0.5) * 100.0) if score_samples else 0.0
        tolerance_10_pct = float(np.mean(diffs_cont > 1.0) * 100.0) if score_samples else 0.0
        severe_misses_cont = float(np.mean(diffs_cont > 1.5) * 100.0) if score_samples else 0.0

        macro_acc = (bool_acc + choice_acc + score_acc) / 3.0
        macro_ece_ew = (bool_ece_ew + choice_ece_ew) / 2.0
        macro_ece_ad = (bool_ece_ad + choice_ece_ad) / 2.0
        macro_brier = (bool_brier + choice_brier) / 2.0

        return {
            "bool": {
                "acc": round(float(bool_acc * 100), 2),
                "ece_equal_width": round(float(bool_ece_ew * 100), 2),
                "ece_adaptive": round(float(bool_ece_ad * 100), 2),
                "brier_score": round(float(bool_brier), 4),
            },
            "choice": {
                "acc": round(float(choice_acc * 100), 2),
                "ece_equal_width": round(float(choice_ece_ew * 100), 2),
                "ece_adaptive": round(float(choice_ece_ad * 100), 2),
                "brier_score": round(float(choice_brier), 4),
            },
            "score": {
                "acc": round(float(score_acc * 100), 2),
                "mae_cont": round(mae_cont, 3),
                "rmse_cont": round(rmse_cont, 3),
                "severe_misses_pct": round(float(severe_misses_cont), 2),
                "tolerance_05_pct": round(float(tolerance_05_pct), 2),
                "tolerance_10_pct": round(float(tolerance_10_pct), 2),
                "off_by_1_pct": round(float(off1_misses), 2),
                "mae_disc": round(mae_disc, 3),
                "mae": round(mae_cont, 3),
            },
            "summary": {
                "macro_acc": round(float(macro_acc * 100), 2),
                "macro_ece_equal_width": round(float(macro_ece_ew * 100), 2),
                "macro_ece_adaptive": round(float(macro_ece_ad * 100), 2),
                "macro_brier": round(float(macro_brier), 4),
                "score_mae_cont": round(mae_cont, 3),
                "score_severe_misses_pct": round(float(severe_misses_cont), 2),
            }
        }

    def export(self, metrics: Optional[Dict[str, Any]] = None):
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. Weights
        heads_path = os.path.join(self.output_dir, "heads.safetensors")
        tensors = {
            "bool_head.linear.weight": np.array(self.bool_head.linear.weight),
            "bool_head.linear.bias": np.array(self.bool_head.linear.bias),
            "choice_head.w_choice.weight": np.array(self.choice_head.w_choice.weight),
            "score_head.w_score.weight": np.array(self.score_head.w_score.weight),
            "score_head.w_score.bias": np.array(self.score_head.w_score.bias),
        }
        save_file(tensors, heads_path)
        print(f"  [Export] Saved weights to {heads_path}")

        # 2. Calibration
        calib_path = os.path.join(self.output_dir, "calibration.json")
        with open(calib_path, "w", encoding="utf-8") as f:
            json.dump(self.calib_data, f, indent=2)
        print(f"  [Export] Saved calibration to {calib_path}")

        # 3. Metadata
        meta_path = os.path.join(self.output_dir, "heads_metadata.json")
        meta = {
            "format_version": "1.0",
            "openjev_version": "2.0.0",
            "hidden_dim": self.hidden_dim,
            "status": "trained_and_calibrated",
            "config": {
                "phase1_loss": self.phase1_loss,
                "phase1_epochs": self.phase1_epochs,
                "phase1_lr": self.phase1_lr,
                "enable_phase2": self.enable_phase2,
                "phase2_epochs": self.phase2_epochs,
                "phase2_lr": self.phase2_lr,
                "phase2_optimizer": self.phase2_optimizer,
                "skip_phase3": self.skip_phase3,
                "seed": self.seed,
            },
            "calibration": self.calib_data,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # 4. Metrics
        if metrics:
            metrics_path = os.path.join(self.output_dir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(f"  [Export] Saved metrics to {metrics_path}")

        # 5. Convergence log
        if self.convergence_records:
            log_path = os.path.join(self.output_dir, "convergence_log.csv")
            fieldnames = ["phase", "epoch", "primitive", "loss", "grad_norm", "opt_mode"]
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.convergence_records)
            print(f"  [Export] Saved convergence log to {log_path} ({len(self.convergence_records)} steps)")

    def train(self, features: Optional[Dict[str, Dict[str, List[Any]]]] = None) -> Dict[str, Any]:
        start_time = time.perf_counter()

        if features is None:
            train_feat = self.extract_features("train")
            val_feat = self.extract_features("val")
            test_feat = self.extract_features("test")
        else:
            train_feat = features["train"]
            val_feat = features["val"]
            test_feat = features.get("test", val_feat)

        # FASE 1: Linear Probing
        self.run_phase1(train_feat)

        # FASE 2: Warm-Start Curriculum Refinement (opcional)
        if self.enable_phase2:
            self.run_phase2(train_feat)

        # FASE 3: Calibración Extrínseca Post-Hoc
        self.run_phase3(val_feat)

        # Evaluación en Test Split
        metrics = self.evaluate(test_feat)
        print("\n[OpenJEV Trainer] === Métricas Finales en Test Split ===")
        print(f"  • Bool Acc:   {metrics['bool']['acc']}%  |  ECE: {metrics['bool']['ece_equal_width']}%  |  Brier: {metrics['bool']['brier_score']}")
        print(f"  • Choice Acc: {metrics['choice']['acc']}%  |  ECE: {metrics['choice']['ece_equal_width']}%  |  Brier: {metrics['choice']['brier_score']}")
        print(f"  • Score Acc:  {metrics['score']['acc']}%  |  Severe Misses (|Δ|≥2): {metrics['score']['severe_misses_pct']}%  |  MAE: {metrics['score']['mae']}")
        print(f"  • Resumen: Macro Acc = {metrics['summary']['macro_acc']}%, Macro ECE = {metrics['summary']['macro_ece_equal_width']}%")

        # Exportar artefactos
        self.export(metrics=metrics)

        elapsed = time.perf_counter() - start_time
        print(f"\n🎉 [OpenJEV Trainer] Pipeline completado con éxito en {elapsed:.2f}s!")
        return metrics

def main():
    parser = argparse.ArgumentParser(description="OpenJEV Supervised Head Training & Calibration CLI")
    parser.add_argument("--model-dir", default="./models/qwen2.5-0.5b", help="Path to model directory")
    parser.add_argument("--dataset-dir", default=None, help="Path to dataset directory (defaults to auto-detected)")
    parser.add_argument("--output-dir", default=None, help="Directory to export heads and calibration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible results")

    # Phase 1
    parser.add_argument("--phase1-loss", choices=["ce", "focal_brier"], default="focal_brier",
                        help="Loss function selector for Phase 1 (ce or focal_brier)")
    parser.add_argument("--phase1-epochs", type=int, default=60, help="Epochs for Phase 1")
    parser.add_argument("--phase1-lr", type=float, default=1e-3, help="Learning rate for Phase 1")

    # Phase 2
    parser.add_argument("--enable-phase2", action="store_true", help="Enable Phase 2 curriculum refinement")
    parser.add_argument("--phase2-epochs", type=int, default=30, help="Epochs for Phase 2")
    parser.add_argument("--phase2-lr", type=float, default=2e-4, help="Learning rate for Phase 2")
    parser.add_argument("--phase2-optimizer", choices=["reset", "continuous"], default="reset",
                        help="Optimizer state handling for Phase 2: 'reset' (cold reinit) or 'continuous' (preserve momentum)")

    # Phase 3
    parser.add_argument("--skip-phase3", action="store_true", help="Skip Phase 3 post-hoc temperature calibration (keep T=1.0)")
    parser.add_argument("--fit-score-temperature", action="store_true", help="Fit temperature for score head")

    # Logging
    parser.add_argument("--log-convergence", action="store_true", help="Record step-by-step loss and grad_norm to convergence_log.csv")

    args = parser.parse_args()

    trainer = HeadTrainer(
        model_dir=args.model_dir,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        phase1_loss=args.phase1_loss,
        phase1_epochs=args.phase1_epochs,
        phase1_lr=args.phase1_lr,
        enable_phase2=args.enable_phase2,
        phase2_epochs=args.phase2_epochs,
        phase2_lr=args.phase2_lr,
        phase2_optimizer=args.phase2_optimizer,
        skip_phase3=args.skip_phase3,
        fit_score_temperature=args.fit_score_temperature,
        log_convergence=args.log_convergence,
    )
    trainer.train()

if __name__ == "__main__":
    main()
