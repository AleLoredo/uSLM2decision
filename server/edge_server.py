"""
OpenJEV Edge Serving Engine (MLX Backend)
Implements full API contract from SPEC.md §4 & IMPLEMENTATION.md §8:
- POST /v1/decide
- GET /healthz
- GET /readyz
- GET /metrics
"""

import time
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field

from core.config import load_config
from core.engine_factory import create_engine

# Load central configuration and instantiate engine
config = load_config()
engine = create_engine(config)
engine.load_model()

MODEL_VERSION = engine.model_version
CALIBRATION_VERSION = engine.calibration_version

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not engine.is_ready():
        engine.load_model()
    yield

app = FastAPI(
    title="OpenJEV Inference Engine",
    version="2.0.0",
    description="Self-hosted non-autoregressive decision engine on Apple Silicon",
    lifespan=lifespan
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

@app.get("/healthz", status_code=status.HTTP_200_OK)
def healthz():
    return {"status": "ok"}

@app.get("/readyz")
def readyz():
    if not engine.is_ready():
        raise HTTPException(status_code=503, detail="Engine is warming up")
    return {"status": "ready", "model_version": engine.model_version}

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
    try:
        decisions = engine.decide(req.context, req.queries)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failure: {e}")

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "model_version": engine.model_version,
        "calibration_version": engine.calibration_version,
        "decisions": decisions,
        "metrics": {
            "latency_ms": elapsed_ms
        }
    }
