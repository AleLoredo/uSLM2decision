#!/usr/bin/env bash
set -e

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
OS="$(uname -s)"
IMAGE_NAME="openjev"

RUNTIME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rust|-r)
      RUNTIME="rust"
      shift
      ;;
    --python|-p)
      RUNTIME="python"
      shift
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

echo "=== Arrancando OpenJEV ==="

# Auto-detección del runtime instalado si no se especificó un flag
# (Comentado temporalmente mientras se completa la reimplementación de Rust)
# if [ -z "$RUNTIME" ]; then
#   if [ -f "./target/release/openjev-mlx" ] && [ ! -d ".venv" ]; then
#     RUNTIME="rust"
#   elif [ -d ".venv" ] && [ ! -f "./target/release/openjev-mlx" ]; then
#     RUNTIME="python"
#   elif [ -f "./target/release/openjev-mlx" ] && [ -d ".venv" ]; then
#     RUNTIME="${OPENJEV_RUNTIME:-python}"
#   fi
# fi
if [ -z "$RUNTIME" ]; then
  RUNTIME="python"
fi

# 1. Rama de Ejecución en Rust
# (Comentada temporalmente - próximamente disponible)
# if [ "$RUNTIME" = "rust" ]; then
#   if [ ! -f "./target/release/openjev-mlx" ]; then
#     echo "❌ Error: El binario './target/release/openjev-mlx' no existe."
#     exit 1
#   fi
#   echo "-> Runtime seleccionado: Rust Nativo (openjev-mlx)"
#   echo "-> Servidor escuchando en http://${HOST}:${PORT}"
#   exec ./target/release/openjev-mlx
# fi
if [ "$RUNTIME" = "rust" ]; then
  echo "⚠️  El runtime nativo de Rust estará disponible próximamente."
  echo "Por el momento se ejecutará el runtime oficial en Python..."
  RUNTIME="python"
fi

# 2. Modo Docker (Linux / Windows con imagen construida)
if [[ "$OS" =~ ^(Linux|MINGW|MSYS|CYGWIN) ]] && command -v docker &> /dev/null && docker image inspect "$IMAGE_NAME" &> /dev/null; then
  echo "-> Modo de ejecución: Contenedor Docker ($IMAGE_NAME)"

  GPU_FLAG=""
  if (command -v nvidia-smi &> /dev/null || command -v nvidia-smi.exe &> /dev/null) && docker info 2>/dev/null | grep -iq "nvidia"; then
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

# 3. Rama de Ejecución en Python (.venv)
if [ -d ".venv" ]; then
  echo "-> Runtime seleccionado: Python (.venv + MLX / PyTorch)"
  # shellcheck disable=SC1091
  source .venv/bin/activate

  export PYTHONPATH="."
  echo "-> Servidor escuchando en http://${HOST}:${PORT}"
  exec uvicorn server.edge_server:app --host "$HOST" --port "$PORT"
fi

# 4. Si no se encontró ningún runtime configurado
echo "❌ Error: No se encontró ningún runtime instalado (ni './target/release/openjev-mlx' ni '.venv')."
echo "Por favor, corre primero el instalador:"
echo "  ./install.sh"
exit 1