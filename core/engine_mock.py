"""
OpenJEV Mock Inference Engine
Provides rapid, deterministic or synthetic decisions for integration tests and CI without requiring model weights.
Conforms to SPEC.md and ARCHITECTURE.md.
"""

import json
import numpy as np
from typing import Dict, Any, Union
from core.engine import BaseInferenceEngine
from core.config import OpenJEVConfig

class MockInferenceEngine(BaseInferenceEngine):
    def __init__(self, config: OpenJEVConfig):
        self.config = config
        self._is_ready = False
        self._model_version = f"mock-{config.model.active_model}"
        self._calibration_version = "calib-mock-v1"

    def load_model(self) -> None:
        self._is_ready = True

    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def calibration_version(self) -> str:
        return self._calibration_version

    def decide(self, state: Union[str, Dict[str, Any], list], questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        answers = {}
        for q_id, q in questions.items():
            q_type = q.get("type")

            if q_type == "noul":
                p_true = float(np.random.uniform(0.1, 0.9))
                answers[q_id] = {
                    "type": "noul",
                    "noul": 1 if p_true >= 0.5 else 0
                }
            elif q_type == "choice":
                criteria = q.get("criteria", {})
                if not criteria:
                    raise ValueError(f"Question '{q_id}' of type 'choice' requires non-empty criteria map")

                opt_ids = list(criteria.keys())
                raw_logits = np.random.normal(0, 1, size=len(opt_ids))
                exp_logits = np.exp(raw_logits - np.max(raw_logits))
                probs = exp_logits / np.sum(exp_logits)
                sel_idx = int(np.argmax(probs))

                answers[q_id] = {
                    "type": "choice",
                    "choice": opt_ids[sel_idx],
                    "confidence": round(float(probs[sel_idx]), 4),
                    "probabilities": {opt: round(float(p), 4) for opt, p in zip(opt_ids, probs)}
                }
            elif q_type == "score":
                criteria = q.get("criteria", [])
                if not criteria or len(criteria) < 2:
                    raise ValueError(f"Question '{q_id}' of type 'score' requires criteria list with at least 2 levels")

                num_levels = len(criteria)
                raw_logits = np.random.normal(0, 1, size=num_levels)
                exp_logits = np.exp(raw_logits - np.max(raw_logits))
                probs = exp_logits / np.sum(exp_logits)

                levels = np.arange(num_levels)
                score = float(np.sum(probs * levels))
                discrete_level = int(np.clip(np.round(score), 0, num_levels - 1))
                confidence = float(probs[discrete_level])

                legend_map = {str(i): desc for i, desc in enumerate(criteria)}
                probs_dict = {str(i): round(float(p), 4) for i, p in enumerate(probs)}

                answers[q_id] = {
                    "type": "score",
                    "score": round(score, 2),
                    "confidence": round(confidence, 4),
                    "legend": legend_map,
                    "probabilities": probs_dict
                }
            else:
                raise ValueError(f"Unsupported question type '{q_type}'")

        state_str = state if isinstance(state, str) else json.dumps(state)
        input_tokens = len(state_str.split()) + sum(len(str(q).split()) for q in questions.values())
        output_tokens = len(json.dumps(answers).split())

        return {
            "answers": answers,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
        }
