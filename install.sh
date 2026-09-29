#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

OS="$(uname -s)"
ARCH="$(uname -m)"

# Procesar flags de línea de comandos para accesos directos
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check|-c|--doctor)
      "$SCRIPT_DIR/scripts/check_env.sh"
      exit 0
      ;;
    --rust|-r)
      "$SCRIPT_DIR/scripts/install_rust.sh"
      exit 0
      ;;
    --python|-p)
      "$SCRIPT_DIR/scripts/install_python.sh"
      exit 0
      ;;
    --model|-m)
      "$SCRIPT_DIR/scripts/setup_model.sh"
      exit 0
      ;;
    --help|-h)
      echo "Uso: ./install.sh [opciones]"
      echo ""
      echo "Opciones:"
      echo "  --python, -p       Instalar directamente el runtime de Python (.venv / Docker)"
      echo "  --rust, -r         Instalar directamente el runtime nativo de Rust (openjev-mlx)"
      echo "  --model, -m        Ejecutar solo el asistente de configuración de modelo"
      echo "  --check, -c        Ejecutar diagnóstico de entorno y compatibilidad (Doctor)"
      echo "  --help, -h         Mostrar esta ayuda"
      exit 0
      ;;
    *)
      echo "Opción no reconocida: $1. Usa --help para ver opciones disponibles."
      exit 1
      ;;
  esac
done

echo "=== Configuración e Instalación de OpenJEV ==="
echo "Sistema detectado: $OS ($ARCH)"
echo ""
echo "============================================================"
echo "           Selección de Runtime de OpenJEV"
echo "============================================================"
echo "¿Qué motor deseas instalar en este equipo?"
echo "  [1] Python (Recomendado: Apple MLX / Metal GPU, o Docker en Linux/Windows)"
echo "  [2] Rust Nativo (Binario compilado independiente openjev-mlx, Apple Silicon)"
echo ""
read -r -p "Selecciona una opción [1/2] (por defecto: 1): " RUNTIME_CHOICE
RUNTIME_CHOICE="${RUNTIME_CHOICE:-1}"

if [ "$RUNTIME_CHOICE" = "2" ]; then
  "$SCRIPT_DIR/scripts/install_rust.sh"
else
  "$SCRIPT_DIR/scripts/install_python.sh"
fi