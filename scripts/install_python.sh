#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OS="$(uname -s)"
ARCH="$(uname -m)"
IMAGE_NAME="openjev"

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

    "$SCRIPT_DIR/setup_model.sh" "Darwin"
    ;;

  Linux*|MINGW*|MSYS*|CYGWIN*)
    # Ejecutar diagnóstico de entorno y cargar variables de estado
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/check_env.sh"

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
        if [ "$IS_WINDOWS_HOST" = true ]; then
          echo " 2. Asegúrate de tener WSL 2 configurado ('wsl --install')."
          echo " 3. En Docker Desktop, activa 'Use the WSL 2 based engine'."
        fi
        echo " 4. Asegúrate de que el daemon esté activo ('docker info')."
        echo " 5. Vuelve a ejecutar: ./install.sh"
        exit 0
      else
        echo "-> Continuando con la instalación nativa local..."
      fi
    else
      echo "-> Docker detectado y activo."
      read -r -p "¿Deseas compilar la imagen Docker para OpenJEV? [S/n]: " DOCKER_RESP
      DOCKER_RESP="${DOCKER_RESP:-s}"

      if [[ "$DOCKER_RESP" =~ ^[sSyY]$ ]]; then
        if [ "$DOCKER_NVIDIA" = false ]; then
          echo "⚠️  Aviso: Soporte de GPU NVIDIA en Docker no detectado."
          if [ "$IS_WINDOWS_HOST" = true ]; then
            echo "   Asegúrate de que Docker Desktop use el backend WSL 2 y tengas drivers NVIDIA actualizados."
          else
            echo "   Asegúrate de tener configurado 'nvidia-container-toolkit' para aceleración por hardware."
          fi
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

    "$SCRIPT_DIR/setup_model.sh" "Linux"
    ;;

  *)
    echo "❌ Error: Sistema operativo no soportado ($OS)."
    exit 1
    ;;
esac
