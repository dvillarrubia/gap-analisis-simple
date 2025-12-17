# ============================================
# Content Gap Analysis V2
# Docker Image - Produccion
# ============================================

FROM python:3.11-slim

# Metadata
LABEL maintainer="SEO Team"
LABEL description="Content Gap Analysis API - Cluster Reinforcement V2"
LABEL version="1.0"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FLASK_APP=clustering_server.py \
    FLASK_ENV=production \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    HF_HOME=/app/.cache/huggingface

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copiar requirements primero (para cache de Docker)
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Crear directorios necesarios (uploads, logs, data, cache numba, cache huggingface)
RUN mkdir -p /app/uploads /app/logs /app/data /tmp/numba_cache /app/.cache/huggingface && \
    chmod 777 /app/uploads /app/logs /app/data /tmp/numba_cache /app/.cache/huggingface

# Pre-descargar modelo de Sentence Transformers (evita descarga en runtime)
# Solo se usa el modelo español 768D por mayor precisión semántica
# HF_HOME ya apunta a /app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; \
    print('Descargando modelo espanol 768D...'); \
    SentenceTransformer('hiiamsid/sentence_similarity_spanish_es'); \
    print('Modelo descargado correctamente')"

# Copiar codigo de la aplicacion
COPY clustering_server.py .

# Copiar configuracion de ejemplo
COPY .env.example .env.example

# Copiar archivo HTML del frontend
COPY cluster-reinforcement-v2.html .
COPY index.html .

# Usuario no-root para seguridad
# Crear usuario y asignar permisos a todos los directorios de la app
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app /tmp/numba_cache
USER appuser

# Puerto de la aplicacion
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Comando de inicio con Gunicorn (produccion)
# Nota: Sin --preload para evitar problemas con numba cache
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "300", \
     "--keep-alive", "5", \
     "--max-requests", "100", \
     "--max-requests-jitter", "20", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "clustering_server:app"]
