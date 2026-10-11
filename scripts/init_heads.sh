#!/usr/bin/env bash
set -e

# OpenJEV Task Heads Initializer for Hugging Face Models
# Adapts an arbitrary downloaded Hugging Face model directory for OpenJEV by generating
# canonical heads.safetensors and heads_metadata.json with exact dimensions, without training.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -z "$1" ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
  echo "Uso: ./scripts/init_heads.sh <directorio_del_modelo> [--force]"
  echo ""
  echo "Ejemplo:"
  echo "  ./scripts/init_heads.sh ./models/qwen2.5-0.5b"
  echo "  ./scripts/init_heads.sh ./models/mi-modelo-hf --force"
  echo ""
  echo "Descripción:"
  echo "  Inspecciona 'config.json' en el directorio del modelo descargado de Hugging Face,"
  echo "  detecta su 'hidden_size' y genera la estructura canónica de cabezales ('heads.safetensors')"
  echo "  para que OpenJEV pueda cargarlo inmediatamente. Este paso NO ejecuta entrenamiento."
  exit 0
fi

MODEL_DIR="$1"
FORCE_FLAG=""
if [ "$2" = "--force" ] || [ "$2" = "-f" ]; then
  FORCE_FLAG="--force"
fi

if [ ! -d "$MODEL_DIR" ]; then
  echo "❌ Error: El directorio '$MODEL_DIR' no existe."
  exit 1
fi

if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "❌ Error: No se encontró 'config.json' en '$MODEL_DIR'."
  echo "   Asegúrate de que sea un modelo válido descargado de Hugging Face."
  exit 1
fi

echo "=== OpenJEV: Inicializador de Task Heads para Modelos Hugging Face ==="
echo "-> Directorio del modelo: $MODEL_DIR"

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

PYTHONPATH=. python3 -m core.model_scanner --model-dir "$MODEL_DIR" --init-heads $FORCE_FLAG

echo "✅ Task heads inicializados correctamente en: $MODEL_DIR"
echo "   El modelo ya es compatible con OpenJEV y puede ser cargado sin entrenamiento previo."
