FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake gcc g++ git curl \
    libmagic1 poppler-utils \
    libreoffice-writer libreoffice-impress libreoffice-calc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download BGE embedding model
ENV FASTEMBED_CACHE_PATH=/app/.fastembed_cache
RUN python -c "\
from fastembed import TextEmbedding; \
print('Downloading BAAI/bge-small-en-v1.5...'); \
list(TextEmbedding('BAAI/bge-small-en-v1.5').embed(['warmup'])); \
print('Embedding model ready.')"

RUN mkdir -p /app/docs

COPY rag_core.py .
COPY main.py .
COPY log_config.json .

# Bake git commit hash into image
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-config", "/app/log_config.json"]