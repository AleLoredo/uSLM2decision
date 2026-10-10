"""
OpenJEV Edge Serving Engine (MLX Backend)
Implements full API contract from SPEC.md §4 & IMPLEMENTATION.md §8:
- POST /v1/decide
- GET /healthz
- GET /readyz
- GET /metrics (Dynamic Prometheus metrics)
Integrated Middlewares:
- Bearer Authentication (IMPLEMENTATION.md §8)
- Hardware Admission & Load Shedding (IMPLEMENTATION.md §8.1)
- Standalone Rate Limiting (IMPLEMENTATION.md §8.1)
- Correlation Trace IDs (IMPLEMENTATION.md §8)
"""

import time
import uuid
import asyncio
from typing import Dict, List, Any, Optional, Union
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field, field_validator
from core.config import load_config
from core.engine_factory import create_engine
from core.batching.types import QueuedRequest, QueueFullError
from core.batching.queue import BatchQueue
from core.batching.dispatcher import BatchDispatcher
from server.metrics import metrics_collector
from server.middleware.auth import BearerAuthMiddleware
from server.middleware.admission import HardwareAdmissionMiddleware
from server.middleware.rate_limiter import StandaloneRateLimiterMiddleware
from server.middleware.trace import RequestTraceMiddleware

# Load central configuration and instantiate engine
config = load_config()
engine = create_engine(config)
engine.load_model()

MODEL_VERSION = engine.model_version
CALIBRATION_VERSION = engine.calibration_version

# Dynamic batching subsystem
batch_queue: Optional[BatchQueue] = None
batch_dispatcher: Optional[BatchDispatcher] = None

def get_batching_system():
    global batch_queue, batch_dispatcher
    if batch_queue is None and config.server.batching.enabled:
        batch_queue = BatchQueue(
            max_batch_size=config.server.batching.max_batch_size,
            max_wait_ms=config.server.batching.max_wait_ms,
            max_queue_depth=config.server.batching.max_queue_depth
        )
        batch_dispatcher = BatchDispatcher(engine=engine, queue=batch_queue)
        batch_dispatcher.start()
    return batch_queue, batch_dispatcher

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not engine.is_ready():
        engine.load_model()
    if config.server.batching.enabled:
        get_batching_system()
    yield
    if batch_dispatcher:
        batch_dispatcher.stop()

app = FastAPI(
    title="OpenJEV Inference Engine",
    version="2.0.0",
    description="Self-hosted non-autoregressive decision engine on Apple Silicon",
    lifespan=lifespan
)

# Attach middlewares (outermost to innermost execution)
app.add_middleware(BearerAuthMiddleware, config=config)
app.add_middleware(StandaloneRateLimiterMiddleware, config=config)
app.add_middleware(HardwareAdmissionMiddleware, config=config)
app.add_middleware(RequestTraceMiddleware)

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
    queries: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]

    @field_validator("queries", mode="before")
    @classmethod
    def normalize_queries(cls, v):
        if isinstance(v, dict):
            normalized = []
            for q_id, q_data in v.items():
                if isinstance(q_data, dict):
                    item = dict(q_data)
                    item.setdefault("id", str(q_id))
                    normalized.append(item)
                else:
                    normalized.append(q_data)
            return normalized
        return v

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
    # Dynamic live Prometheus format per IMPLEMENTATION.md §8
    prometheus_data = metrics_collector.generate_prometheus_text()
    return Response(content=prometheus_data, media_type="text/plain; version=0.0.4")

@app.post("/v1/decide")
async def decide(req: DecideRequest):
    if not req.queries:
        raise HTTPException(status_code=400, detail="Queries array cannot be empty")

    start_time = time.perf_counter()
    b_queue, b_disp = get_batching_system()

    if config.server.batching.enabled and b_queue and b_disp:
        b_disp.start()
        loop = asyncio.get_running_loop()
        req_future = loop.create_future()
        queued_req = QueuedRequest(
            request_id=str(uuid.uuid4()),
            context=req.context,
            queries=req.queries,
            arrival_time=time.time(),
            loop=loop,
            future=req_future
        )
        try:
            b_queue.put(queued_req)
            decisions = await req_future
        except QueueFullError:
            raise HTTPException(
                status_code=503,
                detail="GPU batch queue at capacity",
                headers={"Retry-After": str(config.server.admission.retry_after_seconds)}
            )
        except (ValueError, KeyError) as e:
            for q in req.queries:
                metrics_collector.record_request(
                    primitive=q.get("type", "unknown"),
                    status_code=400,
                    latency_sec=time.perf_counter() - start_time
                )
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            for q in req.queries:
                metrics_collector.record_request(
                    primitive=q.get("type", "unknown"),
                    status_code=500,
                    latency_sec=time.perf_counter() - start_time
                )
            raise HTTPException(status_code=500, detail=f"Inference failure: {e}")
    else:
        try:
            decisions = engine.decide(req.context, req.queries)
        except (ValueError, KeyError) as e:
            for q in req.queries:
                metrics_collector.record_request(
                    primitive=q.get("type", "unknown"),
                    status_code=400,
                    latency_sec=time.perf_counter() - start_time
                )
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            for q in req.queries:
                metrics_collector.record_request(
                    primitive=q.get("type", "unknown"),
                    status_code=500,
                    latency_sec=time.perf_counter() - start_time
                )
            raise HTTPException(status_code=500, detail=f"Inference failure: {e}")

    elapsed_sec = time.perf_counter() - start_time
    elapsed_ms = elapsed_sec * 1000.0

    # Record successful metrics per primitive
    for q in req.queries:
        metrics_collector.record_request(
            primitive=q.get("type", "unknown"),
            status_code=200,
            latency_sec=elapsed_sec,
            gpu_eval_sec=elapsed_sec
        )

    return {
        "model_version": engine.model_version,
        "calibration_version": engine.calibration_version,
        "decisions": decisions,
        "metrics": {
            "latency_ms": elapsed_ms
        }
    }

