#!/usr/bin/env bash
set -e

# OpenJEV Supervised Head Training & Calibration Runner
# Usage: ./scripts/train_heads.sh [--model <dir>] [--dataset <dir>] [--epochs <N>] [--lr <rate>]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_DIR="./models/qwen2.5-0.5b"
DATASET_DIR="./train-dataset"
EPOCHS="60"
LR="1e-3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|-m)
      MODEL_DIR="$2"
      shift 2
      ;;
    --dataset|-d)
      DATASET_DIR="$2"
      shift 2
      ;;
    --epochs|-e)
      EPOCHS="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --help|-h)
      echo "Uso: ./scripts/train_heads.sh [opciones]"
      echo ""
      echo "Opciones:"
      echo "  --model, -m <dir>     Directorio del modelo (por defecto: ./models/qwen2.5-0.5b)"
      echo "  --dataset, -d <dir>   Directorio del dataset (por defecto: ./train-dataset)"
      echo "  --epochs, -e <N>      Épocas de optimización (por defecto: 20)"
      echo "  --lr <rate>           Tasa de aprendizaje (por defecto: 1e-3)"
      echo "  --help, -h            Muestra esta ayuda"
      exit 0
      ;;
    *)
      echo "Opción no reconocida: $1. Usa --help para ver opciones."
      exit 1
      ;;
  esac
done

echo "=== OpenJEV: Entrenamiento Supervisado y Calibración de Cabezales ==="
echo "-> Modelo:  $MODEL_DIR"
echo "-> Dataset: $DATASET_DIR"
echo "-> Épocas:  $EPOCHS"
echo "-> LR:      $LR"

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

PYTHONPATH=. python3 scripts/train_heads.py \
  --model-dir "$MODEL_DIR" \
  --dataset-dir "$DATASET_DIR" \
  --epochs "$EPOCHS" \
  --lr "$LR"
