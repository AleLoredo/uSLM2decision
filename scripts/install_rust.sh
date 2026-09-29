#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OS="$(uname -s)"
ARCH="$(uname -m)"

echo ""
echo "============================================================"
echo "       Instalación del Paquete Nativo de Rust (openjev-mlx)"
echo "============================================================"

if [ "$OS" != "Darwin" ] || [ "$ARCH" != "arm64" ]; then
  echo "❌ Error: El runtime nativo de Rust (openjev-mlx) está optimizado exclusivamente"
  echo "   para macOS Apple Silicon (arm64 / Metal GPU)."
  echo "   Para Linux o Windows con aceleración NVIDIA GPU, por favor selecciona la opción [1] (Python / Docker)."
  exit 1
fi

if ! command -v cargo &> /dev/null; then
  echo "❌ Error: 'cargo' no está instalado en el sistema."
  echo "Para instalar el toolchain oficial de Rust, ejecuta:"
  echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
  exit 1
fi

echo "-> Rust detectado: $(cargo --version)"
echo "-> Compilando el paquete nativo 'openjev-mlx' en modo release..."
cargo build --release

echo ""
echo "============================================================"
echo "       Configuración del Motor Rust en config/engine.toml"
echo "============================================================"
mkdir -p config models/qwen2.5-0.5b

echo "¿Cómo deseas configurar el modo inicial de ejecución en Rust?"
echo "  [1] Modo 'mock' (Pruebas sintéticas y validación de endpoints)"
echo "  [2] Modo 'real' (Inferencia real sobre GPU con tensores safetensors)"
echo ""
read -r -p "Selecciona una opción [1/2] (por defecto: 1): " RUST_MODE_CHOICE
RUST_MODE_CHOICE="${RUST_MODE_CHOICE:-1}"

engine_mode="mock"
if [ "$RUST_MODE_CHOICE" = "2" ]; then
  engine_mode="real"
fi

cat > config/engine.toml <<EOF
# OpenJEV Rust Serving Engine Configuration
# Conforms to SPEC.md §4, ARCHITECTURE.md, and IMPLEMENTATION.md §4 & §8

[server]
host = "0.0.0.0"
port = 8000

[server.auth]
enabled = false
api_keys = ["jev-local-dev-key"]

[batching]
max_batch_size = 16        # requests per eval() call
max_wait_ms = 8            # max time (ms) to wait to accumulate a batch
max_queue_depth = 2048     # requests queued before load shedding

[model]
name = "qwen2.5-0.5b"
mode = "$engine_mode"
base_dir = "./models"
calibration_version = "calib-canonical-v1"
EOF

echo "✅ Archivo 'config/engine.toml' configurado con mode = '$engine_mode'."
echo ""
echo "============================================================"
echo "✅ Instalación del motor Rust completada con éxito."
echo "Binario ejecutable generado en: ./target/release/openjev-mlx"
echo "Para arrancar el servidor ejecuta: ./run.sh"
echo "============================================================"
