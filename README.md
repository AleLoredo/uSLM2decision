# OpenJEV

> **Self-Hostable, Calibrated Non-Autoregressive Decision Engine**

OpenJEV is an open-source inference engine designed to evaluate unstructured context against developer-defined discrete schemas in a **single non-autoregressive forward pass**, producing typed, well-calibrated decisions with explicit confidence bounds.

---

## 🌟 Key Features

- **Typed Decision Primitives:**
  - **`bool` (or `noul`):** Binary true/false decisions (`0` or `1`) with calibrated confidence bounds.
  - **`choice`:** Multi-class discrete routing ($1 \le N \le 255$) with full probability distributions over candidate options.
  - **`score`:** Ordinal rubric scoring with expected values, discrete levels, and distribution variance.
- **Calibrated Decision Alignment:**
  - Trained via **Combined Focal + Brier Score Regularization**, achieving an Expected Calibration Error (**ECE**) of **3.67%** (vs. 14.82% for standard cross-entropy).
  - Built-in **Fast Post-Hoc Recalibration** tools (< 50ms execution on 200 domain samples) bringing Out-of-Distribution ECE down from **18.26% to 3.34%**.
- **Unified Decision Serving:**
  - **Apple Silicon / Metal GPU (Native MLX):** Ultra-low latency on-device decision scoring running on FastAPI (`server/edge_server.py`).
  - **NVIDIA CUDA Container:** Dockerized deployment with GPU passthrough (`Dockerfile`).
  - **Production-Ready Endpoints:** Strict HTTP schema validation (`POST /v1/decide`), health probes (`/healthz`, `/readyz`), and live Prometheus metrics (`/metrics`).
  - *(Note: A native compiled Rust backend is currently under active development).*

---

## 🚀 Supported Model Families

OpenJEV supports multiple model families and tiers out of the box:

| Model Identifier | Architectural Family | Hidden Dimension | Parameters | Role & Performance Target |
|---|---|---|---|---|
| **`qwen2.5-0.5b`** | Llama-like (SwiGLU, RoPE) | $d=896$ | 0.5B (~380 MB) | **Fast Tier:** Ultra-low latency edge decision routing (p50: ~61ms) |
| **`openelm-450m`** | Apple Non-Uniform (Layer-varying GQA & FFN) | $d=1536$ | 0.45B (~872 MB) | **Efficient Edge Tier:** Asymmetric on-device edge scoring (p50: ~93ms) |
| **`gemma-2-2b`** | Google Gemma (Sliding-window, RMSNorm $\sqrt{d}$) | $d=2304$ | 2.0B (~1.4 GB) | **Balanced Tier:** High-capacity semantic reasoning (p50: ~282ms) |

Base model weights are downloaded on demand from Hugging Face into `./models/<model-name>/` via `./scripts/setup_model.sh`.

---

## 🛠️ Quickstart

### Prerequisites

- **macOS** (Apple Silicon M1/M2/M3/M4) or **Linux / Windows (WSL 2 with NVIDIA GPU)**
- Python 3.10+
- Optional: Docker Engine (with NVIDIA Container Toolkit on Linux/WSL 2)

### 1. Installation

Clone the repository and run the automated installer:

```bash
git clone https://github.com/AleLoredo/OpenJEV.git
cd OpenJEV

# Run interactive installer (sets up .venv, dependencies, and model wizard)
./install.sh
```

#### Shortcut Flags & Utilities

```bash
# Run environment diagnostics and check GPU/WSL/Docker status
./install.sh --check        # or ./scripts/check_env.sh

# Directly install Python virtualenv runtime
./install.sh --python       # or ./scripts/install_python.sh

# Run model setup & downloader wizard independently
./install.sh --model        # or ./scripts/setup_model.sh
```

### 2. Configuration

Runtime settings are configured in `config.yaml`:

```yaml
mode: "real"                  # "mock" (fast test runs) or "real" (GPU inference)
model:
  base_dir: "./models"
  active_model: "qwen2.5-0.5b" # "qwen2.5-0.5b", "openelm-450m", or "gemma-2-2b"
  backbone_path: "./models/qwen2.5-0.5b"
  heads_path: "./models/qwen2.5-0.5b/heads.safetensors"
  backend: "mlx"              # "mlx" (Metal GPU) or "torch" (CUDA)
server:
  host: "0.0.0.0"
  port: 8000
```

#### How to Switch to Another Model (e.g., OpenELM or Gemma 2)

To load a different model when running `./run.sh`, you can use either of two options:

##### Option A: Quick Switching via CLI
```bash
# Switch active model to OpenELM-450M:
python3 -m core.config --set-mode real \
  --set-model openelm-450m \
  --set-backbone ./models/openelm-450m \
  --set-heads ./models/openelm-450m/heads.safetensors

# Switch active model to Gemma 2 2B:
python3 -m core.config --set-mode real \
  --set-model gemma-2-2b \
  --set-backbone ./models/gemma-2-2b \
  --set-heads ./models/gemma-2-2b/heads.safetensors

# Switch active model back to Qwen 2.5 0.5B:
python3 -m core.config --set-mode real \
  --set-model qwen2.5-0.5b \
  --set-backbone ./models/qwen2.5-0.5b \
  --set-heads ./models/qwen2.5-0.5b/heads.safetensors
```

##### Option B: Editing `config.yaml`
Modify the `model` block in [`config.yaml`](config.yaml):
```yaml
model:
  base_dir: ./models
  active_model: openelm-450m                # "openelm-450m", "gemma-2-2b", or "qwen2.5-0.5b"
  backbone_path: ./models/openelm-450m
  heads_path: ./models/openelm-450m/heads.safetensors
  calibration_path: ./models/openelm-450m/calibration.json
```

### 3. Running the Server

Launch the server using `run.sh`:

```bash
# Start server using the active configuration
./run.sh

# Or customize host and port
PORT=9000 HOST=127.0.0.1 ./run.sh
```

### 4. Running the Test Suite

```bash
source .venv/bin/activate
pytest tests/ -v
```

> **Note:** Tests requiring real model backbones automatically skip if local weights have not been downloaded yet via `./scripts/setup_model.sh`.

---

## 📊 Training & Evaluation Datasets (`train-dataset/`)

OpenJEV bundles a bilingual (English / Spanish) multi-task dataset partitioned in JSON Lines (`.jsonl`) format under [`train-dataset/`](train-dataset/):

- **`train.jsonl` (1,440 samples / 80% split):** Consumed by [`scripts/train_heads.py`](scripts/train_heads.py) to train decision task heads (`heads.safetensors`) via supervised linear probing while keeping the transformer backbone frozen.
- **`val.jsonl` (180 samples / 10% split):** Evaluated during head training to monitor validation accuracy and fit the **Post-Hoc Temperature Scaler** ([`calibration/post_hoc.py`](calibration/post_hoc.py)), outputting `calibration.json` to minimize Expected Calibration Error (ECE).
- **`test.jsonl` (180 samples / 10% split):** Blind held-out evaluation set consumed by [`tests/test_model_effectiveness.py`](tests/test_model_effectiveness.py) to assert production readiness on unseen data ($\ge 95\%$ accuracy on `bool`/`choice` and bounded ordinal rubric distance).

> **Note:** The live serving engine ([`server/edge_server.py`](server/edge_server.py)) evaluates requests in zero/few-token passes using the compiled weights (`heads.safetensors`) and calibration parameters (`calibration.json`), without querying `.jsonl` files at runtime.

### Esquema para dataset de entrenamiento

Cada línea en los archivos `.jsonl` (`train.jsonl`, `val.jsonl`, `test.jsonl`) representa un registro de entrenamiento supervisado que modela la llamada a la API (`POST /v1/decide`) junto con la etiqueta esperada (*ground truth*):

| Campo | Tipo | Justificación en la API y Entrenamiento |
| :--- | :--- | :--- |
| **`id`** | `string` | Identificador único de muestra (ej. `"rec_00001"` o `"telco_001"`). Permite trazabilidad, reproducibilidad de métricas de pérdida/ECE y depuración en auditorías de datos. |
| **`primitive`** | `string` (`"choice"`, `"score"`, `"noul"`) | Indica al cargador de entrenamiento ([`scripts/train_heads.py`](scripts/train_heads.py)) cuál de los tres cabezales supervisados (`ChoiceHead`, `ScoreHead` o `BoolHead`) debe recibir y optimizar este registro con su respectiva función de pérdida. |
| **`state`** | `string` \| `object` | Representa el contexto o mensaje de entrada a evaluar. Es el texto que se procesa a través del backbone para extraer el vector de contexto $h_{\text{state}}$. Modela situaciones o estados que el modelo evaluará en una sola pasada hacia adelante. |
| **`question`** | `object` | Modela la pregunta declarada tal como se estructura en `POST /v1/decide` ([`SPEC.md` §3 y §4](SPEC.md)): |
| **`question.type`** | `string` | Tipo de la pregunta; debe coincidir con `primitive` para garantizar la validación de esquema en la API. |
| **`question.instructions`** | `string` | Instrucción que guía la decisión. Permite al modelo diferenciar consultas idénticas con criterios distintos (requisito de sensibilidad a criterios de [`ARCHITECTURE.md` §3.2](ARCHITECTURE.md)). |
| **`question.criteria`** | `dict` (para `choice`), `list` (para `score`), `string` (para `noul`, opcional) | **En `choice`:** Un diccionario `{ "<id>": "<descripción>" }` donde cada clave representa una opción y su valor es la definición semántica que se proyecta en el espacio bilineal.<br>**En `score`:** Una lista ordenada de strings `["nivel 0", "nivel 1", ...]` que define la escala ordinal.<br>**En `noul`:** Opcional (clarificación del enunciado verdadero/falso). |
| **`target`** | `string` / `int` | Etiqueta de verdad terreno (*ground truth*) para el entrenamiento supervisado:<br>• Para `choice`: clave exacta de la opción ganadora definida en `criteria` (ej. `"facturacion"`).<br>• Para `score`: índice entero en la rúbrica (ej. `0`, `1`, `2`).<br>• Para `noul`: entero binario `0` (falso) o `1` (verdadero). |

---

## 📡 Sample API Usage

### Request (`POST /v1/decide`)

```bash
curl -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Hi, I have been trying to connect my Stripe account for 3 days and the integration keeps failing. I am losing sales. Please help ASAP.",
    "model": "qwen2.5-0.5b",
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

### Response

```json
{
  "model": "qwen2.5-0.5b",
  "calibration_version": "calib-canonical-v1",
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