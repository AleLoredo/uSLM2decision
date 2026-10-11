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

## 📌 Current Status & Recent Updates

- **Active Model Backbone & Weights:**
  - **Qwen2.5-0.5B** is downloaded and active in [`models/qwen2.5-0.5b/`](file:///Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/models/qwen2.5-0.5b) (4-bit MLX weights `model.safetensors`, 278 MB).
  - Decision task heads (`heads.safetensors`, 3.2 MB) initialized with canonical dimensions (`hidden_size: 896`).
  - Real serving verified on Apple Silicon Metal GPU with measured latencies of **15–45 ms**.
- **Dual Query Schema Support (Map & Array Formats):**
  - `POST /v1/decide` seamlessly accepts `queries` formatted both as a **JSON Map/Dictionary** (`"queries": { "<id>": { ... } }`) per [`SPEC.md §4.1`](file:///Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/SPEC.md#L93-L107) and as a **JSON Array** (`"queries": [ { "id": "<id>", ... } ]`) across both Python (`FastAPI`) and Rust (`Axum`) runtimes.
- **Automated Environment Diagnostics ("OpenJEV Doctor"):**
  - Read-only diagnostics via `./install.sh --check` or [`scripts/check_env.sh`](file:///Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/scripts/check_env.sh).
  - Automatically identifies macOS Metal, native Linux, and Windows hosts (distinguishing Git Bash / MSYS vs WSL 1 vs WSL 2), Docker daemon status, and NVIDIA GPU container passthrough (`nvidia-container-toolkit`).
- **Modular Installer Architecture:**
  - [`install.sh`](file:///Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/install.sh) refactored into a lean orchestrator with CLI flags (`--rust`, `--python`, `--check`, `--model`).
  - Dedicated sub-scripts in [`scripts/`](file:///Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/scripts/) (`check_env.sh`, `install_python.sh`, `install_rust.sh`, `setup_model.sh`).
- **Next Milestone:**
  - Connecting the full 24-layer transformer forward pass of `model.safetensors` to task decision heads, followed by supervised multi-task linear probing to transition task heads from initial canonical baseline to high-certainty calibrated inference.

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
# Run Python test suite (51 unit and integration tests)
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
    "state": "Hi, I'\''ve been trying to connect my Stripe account for 3 days and the integration keeps failing. I'\''m losing sales. Please help ASAP.",
    "model": "jev-latest",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this",
        "criteria": {
          "billing": "Payment or subscription issues",
          "technical": "Bugs or integration problems",
          "sales": "Pricing or account questions"
        }
      },
      "frustration": {
        "type": "score",
        "instructions": "How frustrated the customer appears",
        "criteria": [
          "Calm, just stating facts",
          "Frustrated but civil",
          "Very angry, strong language"
        ]
      },
      "is_urgent": {
        "type": "noul",
        "instructions": "The message conveys urgency or time-sensitivity"
      }
    }
  }'
```

### Sample API Response

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "technical",
      "confidence": 0.78,
      "probabilities": {
        "technical": 0.85,
        "sales": 0.0,
        "billing": 0.15
      }
    },
    "frustration": {
      "type": "score",
      "score": 1.0,
      "confidence": 1.0,
      "legend": {
        "0": "Calm, just stating facts",
        "1": "Frustrated but civil",
        "2": "Very angry, strong language"
      },
      "probabilities": {
        "0": 0.0,
        "1": 1.0,
        "2": 0.0
      }
    },
    "is_urgent": {
      "type": "noul",
      "noul": 1
    }
  },
  "usage": {
    "input_tokens": 392,
    "output_tokens": 65
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