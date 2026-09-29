#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OS="$(uname -s)"
ARCH="$(uname -m)"

IS_WINDOWS=false
WIN_CONTEXT="N/A"
WSL_STATUS="No detectado"
DOCKER_CLI="No instalado"
DOCKER_DAEMON="Inactivo"
NVIDIA_SUPPORT="No disponible"
HOST_NVIDIA=false
DOCKER_AVAILABLE=false
DOCKER_NVIDIA=false

# 1. Detección de Sistema Operativo y Entorno Windows / WSL
case "$OS" in
  MINGW*|MSYS*|CYGWIN*)
    IS_WINDOWS=true
    WIN_CONTEXT="Windows Host (Git Bash / MSYS)"
    ;;
  Linux*)
    if [ -f "/proc/version" ] && grep -qi "microsoft" /proc/version; then
      IS_WINDOWS=true
      if uname -r | grep -qi "wsl2\|microsoft-standard"; then
        WIN_CONTEXT="Windows Subsystem for Linux (WSL 2)"
        WSL_STATUS="Activo (WSL 2 nativo)"
      else
        WIN_CONTEXT="Windows Subsystem for Linux (WSL 1)"
        WSL_STATUS="⚠️ WSL 1 activo (Sin soporte GPU passthrough)"
      fi
    else
      IS_WINDOWS=false
    fi
    ;;
  Darwin*)
    IS_WINDOWS=false
    ;;
  *)
    IS_WINDOWS=false
    ;;
esac

# Si estamos en Git Bash / MSYS en Windows, revisar si wsl.exe existe y qué versión tiene
if [ "$IS_WINDOWS" = true ] && [ "$WSL_STATUS" = "No detectado" ]; then
  if command -v wsl.exe &> /dev/null; then
    wsl_out="$(wsl.exe --status 2>/dev/null | tr -d '\0' || true)"
    if echo "$wsl_out" | grep -qi "version: 2"; then
      WSL_STATUS="Instalado (WSL 2 por defecto)"
    elif echo "$wsl_out" | grep -qi "version: 1"; then
      WSL_STATUS="⚠️ WSL 1 configurado (Se requiere WSL 2 para GPU)"
    else
      wsl_list="$(wsl.exe -l -v 2>/dev/null | tr -d '\0' || true)"
      if echo "$wsl_list" | grep -q " 2 "; then
        WSL_STATUS="Instalado (Distro WSL 2 activa)"
      elif echo "$wsl_list" | grep -q " 1 "; then
        WSL_STATUS="⚠️ Instalado pero en versión 1 (Actualizar a WSL 2)"
      else
        WSL_STATUS="Instalado (Sin distribuciones configuradas)"
      fi
    fi
  else
    WSL_STATUS="❌ No instalado o no disponible en PATH"
  fi
fi

# 2. Detección de Docker CLI y Daemon
if command -v docker &> /dev/null; then
  DOCKER_CLI="Instalado ($(docker --version 2>/dev/null | cut -d',' -f1 || echo 'Docker'))"
  if docker info &> /dev/null; then
    DOCKER_DAEMON="Activo y respondiendo"
    DOCKER_AVAILABLE=true
    if docker info 2>/dev/null | grep -iq "nvidia"; then
      DOCKER_NVIDIA=true
      NVIDIA_SUPPORT="✅ Activo (nvidia-container-toolkit detectado)"
    else
      NVIDIA_SUPPORT="⚠️ No detectado en Docker"
    fi
  else
    DOCKER_DAEMON="❌ Inactivo (Daemon no iniciado o sin permisos)"
  fi
fi

# 3. Detección de GPU física en Host
if command -v nvidia-smi &> /dev/null || command -v nvidia-smi.exe &> /dev/null; then
  HOST_NVIDIA=true
fi

echo "============================================================"
echo "       Diagnóstico del Entorno (OpenJEV Doctor)"
echo "============================================================"
if [ "$OS" = "Darwin" ]; then
  echo "  • Sistema Operativo:   macOS ($ARCH)"
  if [ "$ARCH" = "arm64" ]; then
    echo "  • Aceleración GPU:     Apple Silicon Metal (Compatible MLX)"
  else
    echo "  • Aceleración GPU:     ❌ Intel x86_64 no compatible con Metal MLX"
  fi
elif [ "$IS_WINDOWS" = true ]; then
  echo "  • Host Windows:        SÍ ($WIN_CONTEXT)"
  echo "  • Estado WSL:          $WSL_STATUS"
  echo "  • Docker CLI:          $DOCKER_CLI"
  echo "  • Docker Daemon:       $DOCKER_DAEMON"
  echo "  • Soporte GPU NVIDIA:  $NVIDIA_SUPPORT"
  if [ "$HOST_NVIDIA" = true ] && [ "$DOCKER_NVIDIA" = false ]; then
    echo "  • GPU en Host:         Detectada (nvidia-smi presente en host)"
  fi
else
  echo "  • Sistema Operativo:   Linux Nativo ($ARCH)"
  echo "  • Docker CLI:          $DOCKER_CLI"
  echo "  • Docker Daemon:       $DOCKER_DAEMON"
  echo "  • Soporte GPU NVIDIA:  $NVIDIA_SUPPORT"
  if [ "$HOST_NVIDIA" = true ] && [ "$DOCKER_NVIDIA" = false ]; then
    echo "  • GPU en Host:         Detectada (nvidia-smi presente en host)"
  fi
fi
echo "------------------------------------------------------------"

# 4. Advertencias y guías específicas para Windows y WSL
if [ "$IS_WINDOWS" = true ]; then
  if [[ "$WSL_STATUS" =~ "No instalado" ]] || [[ "$WSL_STATUS" =~ "WSL 1" ]]; then
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

# Exportar variables útiles si el script es ejecutado via source
export IS_WINDOWS_HOST="$IS_WINDOWS"
export DOCKER_AVAILABLE="$DOCKER_AVAILABLE"
export DOCKER_NVIDIA="$DOCKER_NVIDIA"
export HOST_NVIDIA="$HOST_NVIDIA"
