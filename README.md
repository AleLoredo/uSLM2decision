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
- **Multi-Runtime Serving Tier:**
  - **Native Rust Engine (`openjev-mlx`):** Standalone compiled binary on Apple Silicon (Metal GPU) with zero Python overhead, Axum HTTP serving, dynamic batching, native `safetensors` loader, and sub-50ms latency.
  - **Python Edge Engine (Apple Silicon / Metal GPU):** FastAPI server (`server/edge_server.py`) with MLX backend execution.
  - **Containerized GPU Tier (Linux / Windows WSL 2):** Dockerized deployment with NVIDIA CUDA acceleration (`Dockerfile`).
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

- **macOS** (Apple Silicon M1/M2/M3/M4) or **Linux / Windows (WSL 2 required for NVIDIA GPU passthrough)**
- **For Python runtime:** Python 3.10+
- **For Native Rust runtime:** Rust 1.75+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- **For Containerized runtime:** Docker Engine / Docker Desktop (with WSL 2 backend on Windows)

### Automated Setup & Installation

OpenJEV provides an automated cross-platform installation script ([`install.sh`](file:///Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/install.sh)) that inspects your host operating system, diagnoses GPU/Docker/WSL capabilities, and configures the chosen runtime:

```bash
# Clone the repository
git clone https://github.com/AleLoredo/OpenJEV.git
cd OpenJEV

# Run interactive installer (prompts for Python vs Native Rust runtime)
./install.sh
```

#### Shortcut Flags & Modular Scripts

The installer can also be run with non-interactive flags or invoked via modular sub-scripts in [`scripts/`](file:///Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/scripts/):

```bash
# Run environment diagnostics and check GPU/WSL/Docker status (OpenJEV Doctor)
./install.sh --check        # or ./scripts/check_env.sh

# Directly install Native Rust engine (Apple Silicon)
./install.sh --rust         # or ./scripts/install_rust.sh

# Directly install Python runtime (.venv or Docker)
./install.sh --python       # or ./scripts/install_python.sh

# Run model download / setup wizard independently
./install.sh --model        # or ./scripts/setup_model.sh
```

---

### Configuration

OpenJEV supports two runtime engines depending on your deployment target:

1. **Python Engine Configuration ([`config.yaml`](file:///Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/config.yaml)):**
   ```yaml
   mode: "real"                # "mock" (fast dry-runs) or "real" (GPU inference)
   model:
     base_dir: "./models"
     active_model: "qwen2.5-0.5b"
     backbone_path: "./models/qwen2.5-0.5b"
     heads_path: "./models/qwen2.5-0.5b/heads.safetensors"
     backend: "mlx"            # "mlx" (Metal GPU) or "torch" (CUDA)
   server:
     host: "0.0.0.0"
     port: 8000
   ```

2. **Native Rust Engine Configuration ([`config/engine.toml`](file:///Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/config/engine.toml)):**
   ```toml
   [server]
   host = "0.0.0.0"
   port = 8000

   [batching]
   max_batch_size = 16
   max_wait_ms = 8
   max_queue_depth = 2048

   [model]
   name = "qwen2.5-0.5b"
   mode = "real"               # "mock" or "real"
   base_dir = "./models"
   calibration_version = "calib-canonical-v1"
   ```

---

### Running the Server

Use [`run.sh`](file:///Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/run.sh) to launch the server. It automatically detects the installed runtime:

```bash
# Auto-detect and launch the configured server
./run.sh

# Explicitly select runtime:
./run.sh --rust             # Launch compiled Rust binary (openjev-mlx)
./run.sh --python           # Launch Python FastAPI server (.venv)

# Customize host and port:
PORT=9000 HOST=127.0.0.1 ./run.sh
```

---

### Running Tests

```bash
# Run Python test suite (49 unit and integration tests)
source .venv/bin/activate
PYTHONPATH=. pytest tests/ -v

# Run Rust test suite (parity, head math, and block-causal tests)
cargo test --workspace
```

---

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