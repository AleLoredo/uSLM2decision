#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OS_TYPE="${1:-$(uname -s)}"

echo ""
echo "============================================================"
echo "           Configuración del Modelo OpenJEV"
echo "============================================================"
echo "¿Cómo deseas configurar el modo inicial de ejecución?"
echo "  [1] Modo 'mock' (Pruebas rápidas y CI sin descargar pesos)"
echo "  [2] Modo 'real' (Inferencia real no autorregresiva en GPU)"
echo ""
echo "NOTA: Podrás cambiar este modo y las rutas de tus modelos en"
echo "cualquier momento editando el archivo 'config.yaml'."
echo "------------------------------------------------------------"

read -r -p "Selecciona una opción [1/2] (por defecto: 1): " MODE_CHOICE
MODE_CHOICE="${MODE_CHOICE:-1}"

if [ "$MODE_CHOICE" = "2" ]; then
  echo ""
  echo "-> Modo 'real' seleccionado."
  echo "¿Cómo deseas obtener el modelo de referencia Qwen2.5-0.5B?"
  echo "  [D] Descargar automáticamente desde Hugging Face (~380 MB en Mac / ~980 MB en Linux, Recomendado)"
  echo "  [L] Especificar ruta a un modelo local ya existente en tu disco"
  echo "  [M] Cambiar a modo Mock (cancelar descarga de pesos)"
  echo ""
  read -r -p "Selecciona una opción [D/l/m] (por defecto: D): " REAL_OPT
  REAL_OPT="${REAL_OPT:-d}"

  if [[ "$REAL_OPT" =~ ^[mM]$ ]]; then
    python3 -m core.config --set-mode mock
    echo "-> Conmutado a modo 'mock'."
  elif [[ "$REAL_OPT" =~ ^[lL]$ ]]; then
    read -r -p "Indica el nombre o subcarpeta del modelo dentro de ./models/ [qwen2.5-0.5b]: " LOCAL_MODEL_NAME
    LOCAL_MODEL_NAME="${LOCAL_MODEL_NAME:-qwen2.5-0.5b}"
    TARGET_DIR="./models/$LOCAL_MODEL_NAME"
    mkdir -p "$TARGET_DIR"

    while [ ! -f "$TARGET_DIR/heads.safetensors" ]; do
      echo ""
      echo "⚠️  No se encontró 'heads.safetensors' en: $TARGET_DIR"
      echo "   Por favor, coloca los archivos del modelo en:"
      echo "   $TARGET_DIR/"
      echo "     ├── heads.safetensors   (Cabezales de decisión requeridos)"
      echo "     ├── config.json         (Configuración de arquitectura Hugging Face)"
      echo "     └── tokenizer.json      (Tokenizador del modelo)"
      echo ""
      echo "Opciones:"
      echo "  [Enter]  Volver a comprobar tras haber colocado los archivos"
      echo "  [M]      Cambiar a modo Mock y continuar"
      read -r -p "Selecciona una opción o presiona Enter para reintentar: " RETRY_RESP
      if [[ "$RETRY_RESP" =~ ^[mM]$ ]]; then
        python3 -m core.config --set-mode mock
        echo "-> Conmutado a modo 'mock'."
        break
      fi
    done

    if [ -f "$TARGET_DIR/heads.safetensors" ]; then
      python3 -m core.config --set-mode real --set-model "$LOCAL_MODEL_NAME" --set-backbone "$TARGET_DIR" --set-heads "$TARGET_DIR/heads.safetensors"
      echo "✅ Modelo local verificado y configurado en config.yaml"
    fi
  else
    # Opción [D]: Descarga asistida de Hugging Face
    repo_id="mlx-community/Qwen2.5-0.5B-4bit"
    if [ "$OS_TYPE" != "Darwin" ]; then
      repo_id="Qwen/Qwen2.5-0.5B"
    fi

    target_dir="./models/qwen2.5-0.5b"
    mkdir -p "$target_dir"
    echo ""
    echo "-> Descargando '$repo_id' desde Hugging Face hacia $target_dir..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$repo_id', local_dir='$target_dir')"

    echo "-> Inicializando cabezales canónicos heads.safetensors en $target_dir..."
    python3 -m core.model_scanner --model-dir "$target_dir" --init-heads

    python3 -m core.config --set-mode real --set-model qwen2.5-0.5b --set-backbone "$target_dir" --set-heads "$target_dir/heads.safetensors"
    echo "✅ Modelo default Qwen2.5-0.5B descargado y configurado exitosamente."
  fi
else
  python3 -m core.config --set-mode mock
  echo "✅ Modo 'mock' configurado en config.yaml."
fi

echo ""
echo "============================================================"
echo "✅ Configuración del modelo completada con éxito."
echo "Para arrancar el servidor ejecuta: ./run.sh"
echo "============================================================"
