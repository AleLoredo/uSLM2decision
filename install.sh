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
      # "$SCRIPT_DIR/scripts/install_rust.sh"
      echo "⚠️  El runtime nativo de Rust está en desarrollo en la rama 'rsch' y no está disponible en esta versión pública."
      echo "Por favor utiliza el runtime oficial de Python: ./install.sh --python"
      exit 1
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
echo "Instalando runtime oficial de OpenJEV en Python (Apple MLX / Metal GPU)..."
echo "(Nota: El backend en Rust está en desarrollo en la rama 'rsch')"
echo ""

# Ejecución del instalador de Python
"$SCRIPT_DIR/scripts/install_python.sh"