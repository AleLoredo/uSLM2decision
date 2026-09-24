#!/usr/bin/env bash
set -e

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
OS="$(uname -s)"
IMAGE_NAME="openjev"

echo "=== Arrancando OpenJEV ==="

# 1. Modo Docker (Linux con imagen construida)
if [ "$OS" = "Linux" ] && command -v docker &> /dev/null && docker image inspect "$IMAGE_NAME" &> /dev/null; then
  echo "-> Modo de ejecución: Contenedor Docker ($IMAGE_NAME)"

  GPU_FLAG=""
  if command -v nvidia-smi &> /dev/null && docker info 2>/dev/null | grep -iq "nvidia"; then
    echo "-> Aceleración GPU NVIDIA activada (--gpus all)."
    GPU_FLAG="--gpus all"
  else
    echo "-> Ejecutando en CPU (sin aceleración GPU en Docker)."
  fi

  exec docker run --rm -it \
    $GPU_FLAG \
    -p "${PORT}:${PORT}" \
    -e PORT="$PORT" \
    -e HOST="$HOST" \
    "$IMAGE_NAME"
fi

# 2. Modo Nativo / Virtualenv (macOS Apple Silicon o Linux sin Docker)
if [ -d ".venv" ]; then
  echo "-> Modo de ejecución: Entorno virtual (.venv)"
  
  # shellcheck disable=SC1091
  source .venv/bin/activate

  export PYTHONPATH="."
  echo "-> Servidor escuchando en http://${HOST}:${PORT}"
  exec uvicorn server.edge_server:app --host "$HOST" --port "$PORT"
fi

# 3. Si no hay nada configurado
echo "❌ Error: No se encontró ni la imagen Docker '$IMAGE_NAME' ni el entorno virtual '.venv'."
echo "Por favor, corre primero el instalador:"
echo "  ./install.sh"
exit 1