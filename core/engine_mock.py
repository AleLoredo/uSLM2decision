"""
OpenJEV Mock Inference Engine
Provides rapid, deterministic or synthetic decisions for integration tests and CI without requiring model weights.
"""

import numpy as np
from typing import List, Dict, Any
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

    def decide(self, context: str, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        decisions = []
        for idx, q in enumerate(queries):
            q_type = q.get("type")
            q_id = q.get("id", f"q_{idx}")

            if q_type == "bool":
                p_true = float(np.random.uniform(0.1, 0.9))
                val = p_true > 0.5
                conf = p_true if val else 1.0 - p_true
                decisions.append({
                    "id": q_id,
                    "type": "bool",
                    "value": val,
                    "confidence": conf
                })
            elif q_type == "choice":
                options = q.get("options", [])
                if not options:
                    raise ValueError(f"Query '{q_id}' of type 'choice' requires options")

                raw_logits = np.random.normal(0, 1, size=len(options))
                exp_logits = np.exp(raw_logits - np.max(raw_logits))
                probs = exp_logits / np.sum(exp_logits)
                sel_idx = int(np.argmax(probs))

                decisions.append({
                    "id": q_id,
                    "type": "choice",
                    "selected": options[sel_idx]["id"],
                    "confidence": float(probs[sel_idx]),
                    "distribution": {opt["id"]: float(p) for opt, p in zip(options, probs)}
                })
            elif q_type == "score":
                min_val = q.get("min", 1)
                max_val = q.get("max", 5)
                num_levels = max_val - min_val + 1

                raw_logits = np.random.normal(0, 1, size=num_levels)
                exp_logits = np.exp(raw_logits - np.max(raw_logits))
                probs = exp_logits / np.sum(exp_logits)

                levels = np.arange(min_val, max_val + 1)
                score = float(np.sum(probs * levels))
                discrete_level = int(np.clip(np.round(score), min_val, max_val))
                variance = float(np.sum(probs * ((levels - score) ** 2)))
                confidence = float(probs[discrete_level - min_val])

                decisions.append({
                    "id": q_id,
                    "type": "score",
                    "score": score,
                    "discrete_level": discrete_level,
                    "confidence": confidence,
                    "variance": variance,
                    "distribution": [float(p) for p in probs]
                })
            else:
                raise ValueError(f"Unsupported query type '{q_type}'")

        return decisions
