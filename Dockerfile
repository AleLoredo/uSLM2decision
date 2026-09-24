# Imagen base liviana con soporte CUDA 12 y drivers runtime
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Variables de entorno para Python y CUDA
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    APP_MODULE=server.edge_server:app

# Instalar Python, pip y utilitarios mínimos del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-setuptools \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar primero únicamente el archivo de requerimientos para optimizar el cache de Docker
COPY requirements-linux.txt .

# Instalar dependencias de Python (apuntando al índice oficial con CUDA si aplica)
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements-linux.txt

# Copiar el resto del código del repositorio
COPY . .

# Exponer el puerto por defecto (configurable en runtime)
EXPOSE 8000

# Arrancar uvicorn utilizando las variables de entorno pasadas por run.sh
CMD ["sh", "-c", "uvicorn ${APP_MODULE} --host ${HOST} --port ${PORT}"]