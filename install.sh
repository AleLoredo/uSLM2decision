#!/usr/bin/env bash
set -e

OS="$(uname -s)"
ARCH="$(uname -m)"
IMAGE_NAME="openjev"

echo "=== Configuración e Instalación de OpenJEV ==="
echo "Sistema detectado: $OS ($ARCH)"

install_rust_runtime() {
  echo ""
  echo "============================================================"
  echo "       Instalación del Paquete Nativo de Rust (openjev-mlx)"
  echo "============================================================"

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

  local engine_mode="mock"
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
}

configure_openjev_python_wizard() {
  local os_type="$1"
  echo ""
  echo "============================================================"
  echo "           Configuración del Motor OpenJEV (Python)"
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
  echo "✅ Instalación y configuración completadas con éxito."
  echo "Para arrancar el servidor ejecuta: ./run.sh"
  echo "============================================================"
}

diagnose_linux_windows_environment() {
  local is_windows=false
  local win_context="N/A"
  local wsl_status="No detectado"
  local docker_cli="No instalado"
  local docker_daemon="Inactivo"
  local nvidia_support="No disponible"
  local host_nvidia=false

  # 1. Detección de Windows vs Linux Nativo
  case "$OS" in
    MINGW*|MSYS*|CYGWIN*)
      is_windows=true
      win_context="Windows Host (Git Bash / MSYS)"
      ;;
    Linux*)
      if [ -f "/proc/version" ] && grep -qi "microsoft" /proc/version; then
        is_windows=true
        if uname -r | grep -qi "wsl2\|microsoft-standard"; then
          win_context="Windows Subsystem for Linux (WSL 2)"
          wsl_status="Activo (WSL 2 nativo)"
        else
          win_context="Windows Subsystem for Linux (WSL 1)"
          wsl_status="⚠️ WSL 1 activo (Sin soporte GPU passthrough)"
        fi
      else
        is_windows=false
      fi
      ;;
  esac

  # Si estamos en Git Bash / MSYS en Windows, revisar si wsl.exe existe y qué versión tiene
  if [ "$is_windows" = true ] && [ "$wsl_status" = "No detectado" ]; then
    if command -v wsl.exe &> /dev/null; then
      local wsl_out
      wsl_out="$(wsl.exe --status 2>/dev/null | tr -d '\0' || true)"
      if echo "$wsl_out" | grep -qi "version: 2"; then
        wsl_status="Instalado (WSL 2 por defecto)"
      elif echo "$wsl_out" | grep -qi "version: 1"; then
        wsl_status="⚠️ WSL 1 configurado (Se requiere WSL 2 para GPU)"
      else
        local wsl_list
        wsl_list="$(wsl.exe -l -v 2>/dev/null | tr -d '\0' || true)"
        if echo "$wsl_list" | grep -q " 2 "; then
          wsl_status="Instalado (Distro WSL 2 activa)"
        elif echo "$wsl_list" | grep -q " 1 "; then
          wsl_status="⚠️ Instalado pero en versión 1 (Actualizar a WSL 2)"
        else
          wsl_status="Instalado (Sin distribuciones configuradas)"
        fi
      fi
    else
      wsl_status="❌ No instalado o no disponible en PATH"
    fi
  fi

  # 2. Detección de Docker CLI y Daemon
  DOCKER_AVAILABLE=false
  DOCKER_NVIDIA=false

  if command -v docker &> /dev/null; then
    docker_cli="Instalado ($(docker --version 2>/dev/null | cut -d',' -f1 || echo 'Docker'))"
    if docker info &> /dev/null; then
      docker_daemon="Activo y respondiendo"
      DOCKER_AVAILABLE=true
      if docker info 2>/dev/null | grep -iq "nvidia"; then
        DOCKER_NVIDIA=true
        nvidia_support="✅ Activo (nvidia-container-toolkit detectado)"
      else
        nvidia_support="⚠️ No detectado en Docker"
      fi
    else
      docker_daemon="❌ Inactivo (Daemon no iniciado o sin permisos)"
    fi
  fi

  # 3. Detección de GPU física en Host
  if command -v nvidia-smi &> /dev/null || command -v nvidia-smi.exe &> /dev/null; then
    host_nvidia=true
  fi

  echo ""
  echo "============================================================"
  echo "       Diagnóstico del Entorno (Windows / Docker / GPU)"
  echo "============================================================"
  if [ "$is_windows" = true ]; then
    echo "  • Host Windows:        SÍ ($win_context)"
    echo "  • Estado WSL:          $wsl_status"
  else
    echo "  • Sistema Operativo:   Linux Nativo ($ARCH)"
  fi
  echo "  • Docker CLI:          $docker_cli"
  echo "  • Docker Daemon:       $docker_daemon"
  echo "  • Soporte GPU NVIDIA:  $nvidia_support"
  if [ "$host_nvidia" = true ] && [ "$DOCKER_NVIDIA" = false ]; then
    echo "  • GPU en Host:         Detectada (nvidia-smi presente)"
  fi
  echo "------------------------------------------------------------"

  # 4. Advertencias y guías específicas para Windows y WSL
  if [ "$is_windows" = true ]; then
    if [[ "$wsl_status" =~ "No instalado" ]] || [[ "$wsl_status" =~ "WSL 1" ]]; then
      echo ""
      echo "⚠️  NOTA CRÍTICA PARA USUARIOS DE WINDOWS:"
      echo "   Para ejecutar contenedores Linux con aceleración GPU NVIDIA en Docker,"
      echo "   Windows requiere obligatoriamente el backend WSL 2 (Windows Subsystem for Linux 2)."
      echo "   El backend clásico basado en Hyper-V NO permite exponer la GPU al contenedor."
      echo ""
      echo "   Pasos recomendados para habilitar WSL 2:"
      echo "   1. Abre PowerShell como Administrador y ejecuta:"
      echo "        wsl --install"
      echo "   2. Reinicia el equipo si Windows lo solicita."
      echo "   3. Abre Docker Desktop -> Settings -> General:"
      echo "        Asegúrate de marcar 'Use the WSL 2 based engine'."
      echo "   4. Abre Docker Desktop -> Settings -> Resources -> WSL Integration:"
      echo "        Habilita la integración con tu distribución de Linux (ej. Ubuntu)."
      echo "------------------------------------------------------------"
    fi
  fi

  # Exportar variable de host windows para uso en el caller
  IS_WINDOWS_HOST="$is_windows"
}

install_python_runtime() {
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

      configure_openjev_python_wizard "Darwin"
      ;;

    Linux*|MINGW*|MSYS*|CYGWIN*)
      diagnose_linux_windows_environment

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

      configure_openjev_python_wizard "Linux"
      ;;

    *)
      echo "❌ Error: Sistema operativo no soportado ($OS)."
      exit 1
      ;;
  esac
}

echo ""
echo "============================================================"
echo "           Selección de Runtime de OpenJEV"
echo "============================================================"
echo "¿Qué motor deseas instalar en este equipo?"
echo "  [1] Python (Recomendado: Apple MLX / Metal GPU, instalación instantánea)"
echo "  [2] Rust Nativo (Binario compilado independiente openjev-mlx, máxima concurrencia)"
echo ""
read -r -p "Selecciona una opción [1/2] (por defecto: 1): " RUNTIME_CHOICE
RUNTIME_CHOICE="${RUNTIME_CHOICE:-1}"

if [ "$RUNTIME_CHOICE" = "2" ]; then
  install_rust_runtime
else
  install_python_runtime
fi