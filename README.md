# OpenJEV

> **Self-Hostable, Calibrated Non-Autoregressive Decision Engine**

OpenJEV is an open-source inference engine designed to evaluate unstructured context against developer-defined discrete schemas in a **single non-autoregressive forward pass**, producing typed, well-calibrated decisions with explicit confidence bounds.

---

## 🌟 Key Features

- **Typed Decision Primitives:**
  - `bool`: Binary decisions with calibrated confidence and explicit tie-breaking rules.
  - `choice`: Multi-class discrete choices ($1 \le N \le 255$) with full probability distributions over candidate options.
  - `score`: Ordinal rubric scoring with expected values, discrete levels, and distribution variance.
- **Calibrated Alignment:**
  - Trained via **Combined Focal + Brier Score Regularization**, achieving an Expected Calibration Error (**ECE**) of **3.67%** (vs 14.82% for standard cross-entropy).
  - Built-in **Fast Post-Hoc Recalibration** tools (<50ms execution on 200 domain samples) bringing Out-of-Distribution ECE down from **18.26% to 3.34%**.
- **Dual Serving Engines:**
  - **Edge Tier (Apple Silicon / Metal GPU):** Self-hosted FastAPI server (`server/edge_server.py`) delivering **sub-100ms** latency at 256 tokens for 0.5B model tier with thread-safe MLX backend execution.
  - **Cloud Tier (vLLM Prefix-Caching):** Parallel non-autoregressive query evaluation sharing context KV-caches (`server/vllm_prefix_serving.py`).
- **Strict Normative Specification (`SPEC.md`):**
  - Fully compliant HTTP API contract (`POST /v1/decide`, `/healthz`, `/readyz`, `/metrics`).
  - Structured error formats, strict schema validation, version tracking (`spec_version`, `model_version`, `calibration_version`).

---

## 📚 Documentation Map

1. **[`SPEC.md`](file:///Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/SPEC.md):** Normative API contract and schema specification.
2. **[`ARCHITECTURE.md`](file:///Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/ARCHITECTURE.md):** Mathematical formulation of block-causal attention, heads, and loss functions.
3. **[`ROADMAP.md`](file:///Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/ROADMAP.md):** Project phase completion history and hardware benchmarks.
4. **[`DATASET_CARD.md`](file:///Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/DATASET_CARD.md):** Provenance audit and open license compliance.
5. **[`CALIBRATION_ABLATION.md`](file:///Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/CALIBRATION_ABLATION.md):** Empirical loss shaping ablation report.
6. **[`IMPLEMENTATION.md`](file:///Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/IMPLEMENTATION.md):** Serving stack details for Apple Silicon and Rust/MLX bindings.
7. **[`AGENTS.md`](file:///Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/AGENTS.md):** Developer conventions and repository guidelines.

---

## 🛠️ Quickstart

### Prerequisites

- macOS (Apple Silicon M1/M2/M3/M4) or Linux
- Python 3.10+
- PyTorch / MLX / FastAPI dependencies

### Installation

```bash
# Clone the repository
git clone https://github.com/AleLoredo/OpenJEV.git
cd OpenJEV

# Activate virtual environment
source .venv/bin/activate

# Run full test suite
PYTHONPATH=. pytest tests/ -v
```

### Running the Edge Server

```bash
# Start edge server on port 8000
PYTHONPATH=. python3 -m uvicorn server.edge_server:app --reload --port 8000
```

### Sample API Request

```bash
curl -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "context": "Customer requested account cancellation due to price increase.",
    "queries": {
      "is_churn_risk": {
        "type": "bool",
        "criteria": "Does customer express churn intent?"
      },
      "urgency_score": {
        "type": "score",
        "min": 1,
        "max": 5,
        "rubric": "1=low, 5=critical urgency"
      }
    }
  }'
```

### Sample API Response

```json
{
  "spec_version": "2.0.0",
  "model_version": "openjev-0.5b-v1",
  "calibration_version": "calib-focal-brier-v1",
  "latency_ms": 14.2,
  "usage": {
    "input_tokens": 18
  },
  "decisions": {
    "is_churn_risk": {
      "value": true,
      "confidence": 0.942
    },
    "urgency_score": {
      "score": 4.21,
      "discrete_level": 4,
      "confidence": 0.885,
      "variance": 0.35,
      "distribution": [0.01, 0.02, 0.08, 0.54, 0.35]
    }
  }
}
```

---

## 🧪 Benchmark Summary (Apple M3, 0.5B Model)

| Context Length | Batch Size | Latency (p50) | Latency (p99) | ECE (Calibrated) |
|---|---|---|---|---|
| 256 tokens | 1 | 95.2 ms | 101.3 ms | **3.67%** |
| 512 tokens | 1 | 208.8 ms | 213.5 ms | **3.67%** |
| 1024 tokens | 1 | 527.1 ms | 545.6 ms | **3.67%** |

---

## 📜 License

Apache-2.0 License.