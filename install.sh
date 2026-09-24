#!/usr/bin/env bash
set -e

OS="$(uname -s)"
ARCH="$(uname -m)"
IMAGE_NAME="openjev"

echo "=== Configuración e Instalación de OpenJEV ==="
echo "Sistema detectado: $OS ($ARCH)"

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
      pip install fastapi "uvicorn[standard]" pydantic numpy mlx pytest
    fi

    echo "✅ Instalación nativa en macOS completada con éxito."
    echo "Para arrancar el servidor ejecuta: ./run.sh"
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
      pip install fastapi "uvicorn[standard]" pydantic numpy torch pytest
    fi

    echo "✅ Instalación local completada con éxito."
    echo "Para arrancar el servidor ejecuta: ./run.sh"
    ;;

  *)
    echo "❌ Error: Sistema operativo no soportado ($OS)."
    exit 1
    ;;
esac