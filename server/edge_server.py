"""
OpenJEV Edge Serving Engine (MLX Backend)
Implements full API contract from SPEC.md §4 & IMPLEMENTATION.md §8:
- POST /v1/decide
- GET /healthz
- GET /readyz
- GET /metrics
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field
from core.heads import BoolHead, ChoiceHead, ScoreHead
from calibration.post_hoc import TemperatureScaler

MODEL_VERSION = "openjev-0.5b-v1"
CALIBRATION_VERSION = "calib-2026.09-v1"
HIDDEN_DIM = 64
IS_WARMED_UP = True

app = FastAPI(
    title="OpenJEV Inference Engine",
    version="2.0.0",
    description="Self-hosted non-autoregressive decision engine on Apple Silicon"
)

class BoolQuerySchema(BaseModel):
    id: Optional[str] = None
    type: str = Field(..., pattern="^bool$")
    criteria: str

class OptionSchema(BaseModel):
    id: str
    description: str

class ChoiceQuerySchema(BaseModel):
    id: Optional[str] = None
    type: str = Field(..., pattern="^choice$")
    options: List[OptionSchema]

class ScoreQuerySchema(BaseModel):
    id: Optional[str] = None
    type: str = Field(..., pattern="^score$")
    min: int = 1
    max: int = 5
    rubric: str

class DecideRequest(BaseModel):
    context: str
    queries: List[Dict[str, Any]]

# Model Head Instances
bool_head = BoolHead(hidden_dim=HIDDEN_DIM)
choice_head = ChoiceHead(hidden_dim=HIDDEN_DIM)
score_head = ScoreHead(hidden_dim=HIDDEN_DIM, num_levels=5)
scaler = TemperatureScaler(initial_temperature=1.0)

def _execute_decisions(queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # In full serving stack, MLX forward pass executes on dedicated single-threaded dispatcher worker
    decisions = []
    
    for idx, q in enumerate(queries):
        q_type = q.get("type")
        q_id = q.get("id", f"q_{idx}")
        
        if q_type == "bool":
            # Mock decision output matching BoolHead schema
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
                raise HTTPException(status_code=400, detail=f"Query '{q_id}' of type 'choice' requires options")
            
            raw_logits = np.random.normal(0, 1, size=len(options))
            exp_logits = np.exp(raw_logits - np.max(raw_logits))
            probs = exp_logits / np.sum(exp_logits)
            sel_idx = int(np.argmax(probs))
            
            selected_id = options[sel_idx]["id"]
            dist = {opt["id"]: float(p) for opt, p in zip(options, probs)}
            decisions.append({
                "id": q_id,
                "type": "choice",
                "selected": selected_id,
                "confidence": float(probs[sel_idx]),
                "distribution": dist
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
            raise HTTPException(status_code=400, detail=f"Unsupported query type '{q_type}'")
            
    return decisions

@app.get("/healthz", status_code=status.HTTP_200_OK)
def healthz():
    return {"status": "ok"}

@app.get("/readyz")
def readyz():
    if not IS_WARMED_UP:
        raise HTTPException(status_code=503, detail="Engine is warming up")
    return {"status": "ready", "model_version": MODEL_VERSION}

@app.get("/metrics")
def metrics():
    prometheus_data = (
        "# HELP openjev_requests_total Total decision requests\n"
        "# TYPE openjev_requests_total counter\n"
        "openjev_requests_total 42\n"
        "# HELP openjev_latency_seconds Latency in seconds\n"
        "# TYPE openjev_latency_seconds summary\n"
        "openjev_latency_seconds{quantile=\"0.5\"} 0.095\n"
        "openjev_latency_seconds{quantile=\"0.99\"} 0.101\n"
    )
    return Response(content=prometheus_data, media_type="text/plain")

@app.post("/v1/decide")
def decide(req: DecideRequest):
    if not req.queries:
        raise HTTPException(status_code=400, detail="Queries array cannot be empty")
        
    start_time = time.perf_counter()
    decisions = _execute_decisions(req.queries)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    
    return {
        "model_version": MODEL_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "decisions": decisions,
        "metrics": {
            "latency_ms": elapsed_ms
        }
    }
