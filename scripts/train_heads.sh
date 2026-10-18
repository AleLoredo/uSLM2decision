#!/usr/bin/env bash
set -e

# OpenJEV Supervised Head Training & Calibration Runner
# Usage: ./scripts/train_heads.sh [options]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_DIR="./models/qwen2.5-0.5b"
DATASET_DIR=""
OUTPUT_DIR=""
EPOCHS="60"
LR="1e-3"
PHASE1_LOSS="focal_brier"
ENABLE_PHASE2=""
PHASE2_EPOCHS="30"
PHASE2_LR="2e-4"
PHASE2_OPT="reset"
SKIP_PHASE3=""
LOG_CONVERGENCE=""
SEED="42"

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
    --output|-o|--output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --epochs|-e|--phase1-epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --lr|--phase1-lr)
      LR="$2"
      shift 2
      ;;
    --phase1-loss)
      PHASE1_LOSS="$2"
      shift 2
      ;;
    --enable-phase2)
      ENABLE_PHASE2="--enable-phase2"
      shift 1
      ;;
    --phase2-epochs)
      PHASE2_EPOCHS="$2"
      shift 2
      ;;
    --phase2-lr)
      PHASE2_LR="$2"
      shift 2
      ;;
    --phase2-optimizer)
      PHASE2_OPT="$2"
      shift 2
      ;;
    --skip-phase3)
      SKIP_PHASE3="--skip-phase3"
      shift 1
      ;;
    --log-convergence)
      LOG_CONVERGENCE="--log-convergence"
      shift 1
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --help|-h)
      echo "Uso: ./scripts/train_heads.sh [opciones]"
      echo ""
      echo "Opciones:"
      echo "  --model, -m <dir>          Directorio del modelo (por defecto: ./models/qwen2.5-0.5b)"
      echo "  --dataset, -d <dir>        Directorio del dataset (auto-detectado si no se indica)"
      echo "  --output, -o <dir>         Directorio de salida de pesos y métricas"
      echo "  --phase1-loss <loss>       Pérdida en Fase 1: 'ce' o 'focal_brier' (def: focal_brier)"
      echo "  --epochs, -e <N>           Épocas de Fase 1 (por defecto: 60)"
      echo "  --lr <rate>                Learning rate de Fase 1 (por defecto: 1e-3)"
      echo "  --enable-phase2            Activa refinamiento warm-start (Fase 2)"
      echo "  --phase2-epochs <N>        Épocas de Fase 2 (por defecto: 30)"
      echo "  --phase2-lr <rate>         Learning rate de Fase 2 (por defecto: 2e-4)"
      echo "  --phase2-optimizer <mode>  'reset' (cold) o 'continuous' (preserve momentum) (def: reset)"
      echo "  --skip-phase3              Omite calibración post-hoc (deja T=1.0 neutro)"
      echo "  --log-convergence          Exporta convergence_log.csv con pérdidas y grad_norm por paso"
      echo "  --seed <N>                 Semilla aleatoria fija (def: 42)"
      echo "  --help, -h                 Muestra esta ayuda"
      exit 0
      ;;
    *)
      echo "Opción no reconocida: $1. Usa --help para ver opciones."
      exit 1
      ;;
  esac
done

echo "=== OpenJEV: Entrenamiento Desacoplado y Calibración de Cabezales ==="
echo "-> Modelo:         $MODEL_DIR"
echo "-> Fase 1 Loss:    $PHASE1_LOSS"
echo "-> Fase 1 Épocas:  $EPOCHS | LR: $LR"
if [ -n "$ENABLE_PHASE2" ]; then
  echo "-> Fase 2:         Activada (Épocas: $PHASE2_EPOCHS, LR: $PHASE2_LR, Opt: $PHASE2_OPT)"
else
  echo "-> Fase 2:         Desactivada"
fi
if [ -n "$SKIP_PHASE3" ]; then
  echo "-> Fase 3:         Desactivada (T=1.0)"
else
  echo "-> Fase 3:         Activada (Ajuste de Temperatura Post-Hoc)"
fi

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

ARGS=(
  --model-dir "$MODEL_DIR"
  --phase1-loss "$PHASE1_LOSS"
  --phase1-epochs "$EPOCHS"
  --phase1-lr "$LR"
  --phase2-epochs "$PHASE2_EPOCHS"
  --phase2-lr "$PHASE2_LR"
  --phase2-optimizer "$PHASE2_OPT"
  --seed "$SEED"
)

if [ -n "$DATASET_DIR" ]; then
  ARGS+=(--dataset-dir "$DATASET_DIR")
fi
if [ -n "$OUTPUT_DIR" ]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi
if [ -n "$ENABLE_PHASE2" ]; then
  ARGS+=($ENABLE_PHASE2)
fi
if [ -n "$SKIP_PHASE3" ]; then
  ARGS+=($SKIP_PHASE3)
fi
if [ -n "$LOG_CONVERGENCE" ]; then
  ARGS+=($LOG_CONVERGENCE)
fi

PYTHONPATH=. python3 scripts/train_heads.py "${ARGS[@]}"
