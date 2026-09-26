#!/usr/bin/env bash
set -e

OS="$(uname -s)"
ARCH="$(uname -m)"
IMAGE_NAME="openjev"

echo "=== Configuración e Instalación de OpenJEV ==="
echo "Sistema detectado: $OS ($ARCH)"

configure_openjev_wizard() {
  local os_type="$1"
  echo ""
  echo "============================================================"
  echo "           Configuración del Motor OpenJEV"
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
      local repo_id="mlx-community/Qwen2.5-0.5B-4bit"
      if [ "$os_type" != "Darwin" ]; then
        repo_id="Qwen/Qwen2.5-0.5B"
      fi

      local target_dir="./models/qwen2.5-0.5b"
      mkdir -p "$target_dir"
      echo ""
      echo "-> Descargando '$repo_id' desde Hugging Face hacia $target_dir..."
      huggingface-cli download "$repo_id" --local-dir "$target_dir"

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
  echo "✅ Instalación y configuración completadas con éxito."
  echo "Para arrancar el servidor ejecuta: ./run.sh"
  echo "============================================================"
}

case "$OS" in
  Darwin)
    if [ "$ARCH" != "arm64" ]; then
      echo "❌ Error: macOS en arquitectura '$ARCH' no es compatible. Se requiere Apple Silicon (arm64) para MLX."
      exit 1
    fi

    echo "-> Entorno detectado: macOS Apple Silicon"
    echo "-> Configurando entorno local nativo (.venv + MLX Metal GPU)..."

    if ! command -v python3 &> /dev/null; then
      echo "❌ Error: python3 no está instalado en el sistema."
      exit 1
    fi

    if [ ! -d ".venv" ]; then
      echo "-> Creando virtualenv en .venv..."
      python3 -m venv .venv
    fi

    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install --upgrade pip

    if [ -f "requirements-macos.txt" ]; then
      echo "-> Instalando desde requirements-macos.txt..."
      pip install -r requirements-macos.txt
    else
      echo "-> requirements-macos.txt no encontrado. Usando dependencias por defecto..."
      pip install fastapi "uvicorn[standard]" pydantic numpy mlx pytest pyyaml safetensors huggingface_hub transformers
    fi

    configure_openjev_wizard "Darwin"
    ;;

  Linux*|MINGW*|MSYS*|CYGWIN*)
    echo "-> Entorno detectado: Linux / Windows"

    # Verificar disponibilidad de Docker (instalación y daemon activo)
    DOCKER_AVAILABLE=false
    if command -v docker &> /dev/null && docker info &> /dev/null; then
      DOCKER_AVAILABLE=true
    fi

    # Si Docker NO está disponible, promptear al usuario
    if [ "$DOCKER_AVAILABLE" = false ]; then
      echo ""
      echo "⚠️  Docker no está instalado o el servicio no se encuentra en ejecución."
      echo "Para aprovechar el aislamiento de contenedores y aceleración de GPU empaquetada, se recomienda Docker."
      echo ""
      read -r -p "¿Deseas pausar aquí para instalar/iniciar Docker y reintentar? [s/N]: " PAUSE_RESP
      PAUSE_RESP="${PAUSE_RESP:-n}"

      if [[ "$PAUSE_RESP" =~ ^[sSyY]$ ]]; then
        echo ""
        echo "Por favor:"
        echo " 1. Instala o inicia Docker Desktop / Docker Engine."
        echo " 2. Asegúrate de que el daemon esté activo ('docker info')."
        echo " 3. Vuelve a ejecutar: ./install.sh"
        exit 0
      else
        echo "-> Continuando con la instalación nativa local..."
      fi
    else
      # Docker está disponible: preguntar si compilar imagen
      echo "-> Docker detectado y activo."
      read -r -p "¿Deseas compilar la imagen Docker para OpenJEV? [S/n]: " DOCKER_RESP
      DOCKER_RESP="${DOCKER_RESP:-s}"

      if [[ "$DOCKER_RESP" =~ ^[sSyY]$ ]]; then
        if ! docker info 2>/dev/null | grep -iq "nvidia"; then
          echo "⚠️  Aviso: Soporte de GPU NVIDIA en Docker no detectado."
          echo "   Asegúrate de tener configurado 'nvidia-container-toolkit' para aceleración por hardware."
        fi

        echo "-> Compilando imagen Docker '$IMAGE_NAME'..."
        docker build -t "$IMAGE_NAME" .
        echo "✅ Imagen Docker construida con éxito."
        echo "Para arrancar el servidor ejecuta: ./run.sh"
        exit 0
      fi
    fi

    # Flujo de instalación local nativa (.venv + PyTorch CUDA)
    echo "-> Configurando entorno virtual local (.venv + PyTorch CUDA)..."

    if ! command -v python3 &> /dev/null; then
      echo "❌ Error: python3 no está instalado en el sistema."
      exit 1
    fi

    if [ ! -d ".venv" ]; then
      echo "-> Creando virtualenv en .venv..."
      python3 -m venv .venv
    fi

    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install --upgrade pip

    if [ -f "requirements-linux.txt" ]; then
      echo "-> Instalando desde requirements-linux.txt..."
      pip install -r requirements-linux.txt
    else
      echo "-> requirements-linux.txt no encontrado. Usando dependencias por defecto..."
      pip install fastapi "uvicorn[standard]" pydantic numpy torch pytest pyyaml safetensors huggingface_hub transformers
    fi

    configure_openjev_wizard "Linux"
    ;;

  *)
    echo "❌ Error: Sistema operativo no soportado ($OS)."
    exit 1
    ;;
esac