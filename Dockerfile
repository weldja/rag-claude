FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    gcc \
    g++ \
    git \
    curl \
    libmagic1 \
    poppler-utils \
    libreoffice-writer \
    libreoffice-impress \
    libreoffice-calc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the BGE embedding model into the image layer.
# This means first-time Init is instant — no download on the customer's machine.
ENV FASTEMBED_CACHE_PATH=/app/.fastembed_cache
RUN python -c "\
from fastembed import TextEmbedding; \
print('Downloading BAAI/bge-small-en-v1.5 ...'); \
list(TextEmbedding('BAAI/bge-small-en-v1.5').embed(['warmup'])); \
print('Embedding model ready.')"

RUN mkdir -p /app/docs

COPY rag_pipeline.py .
COPY config.yaml .

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "rag_pipeline.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]
