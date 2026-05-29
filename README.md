# RAG — Claude API Edition

Ask questions across your business documents. Uses **Claude API** for LLM responses and **BAAI/bge-small-en-v1.5** (local HuggingFace) for embeddings — so document indexing never leaves your machine.

## Quick Start

```bash
# 1. Clone
git clone <your-repo> && cd rag-claude

# 2. Configure
cp .env.example .env
# Add your Anthropic API key:
#   ANTHROPIC_API_KEY=sk-ant-...
# Optionally change CLAUDE_MODEL (see .env.example for options)

# 3. Start services  (no model download needed — much faster than Ollama)
docker compose up -d

# 4. Open the UI
open http://localhost:8501
```

Drop your documents into the `docs/` folder, click **Init** in the sidebar, and start chatting.

---

## Supported Document Types

| Format | Extension |
|--------|-----------|
| PDF    | `.pdf`    |
| Word   | `.docx` `.doc` |
| PowerPoint | `.pptx` `.ppt` |
| Plain text | `.txt` |
| Excel  | `.xlsx` `.xls` |
| CSV    | `.csv`    |
| RTF    | `.rtf`    |

---

## Claude Models

Set `CLAUDE_MODEL` in `.env`:

| Model | Speed | Quality | Cost (in/out per MTok) |
|-------|-------|---------|------------------------|
| `claude-haiku-4-5-20251001` | ⚡ Fast | Good | $0.80 / $4.00 |
| `claude-sonnet-4-6` | Medium | Great | $3.00 / $15.00 |
| `claude-opus-4-6` | Slower | Best | $15.00 / $75.00 |

The sidebar shows a **live cost meter** — total tokens used and spend for the current session.

---

## Architecture

```
docs/  →  Loaders  →  Chunker (1000/100)  →  BGE Embeddings (local)
                                                      ↓
User → Streamlit UI → Retriever ← PGVector (PostgreSQL)
              ↓
        Anthropic Claude API
```

Embeddings are computed **locally** (no API cost). Only the final answer generation calls the Claude API.

---

## Key Settings (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `CLAUDE_MODEL` | `claude-haiku-4-5-20251001` | Claude model to use |
| `MAX_TOKENS` | `1024` | Max tokens in LLM response |
| `TEMPERATURE` | `0.1` | LLM temperature (0 = deterministic) |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `RETRIEVAL_K` | `4` | Chunks returned per query |
| `CACHE_TTL` | `86400` | Query cache TTL (seconds); 0 = off |

---

## CLI Usage

```bash
# Index documents
docker compose run --rm app python rag_pipeline.py --mode=index

# Ask a question
docker compose run --rm app python rag_pipeline.py --mode=query --query="What is our refund policy?"

# Check status
docker compose run --rm app python rag_pipeline.py --mode=status
```

---

## Services

| Service | Port | Description |
|---------|------|-------------|
| `app`   | 8501 | Streamlit UI |
| `db`    | 5432 | PostgreSQL + pgvector |
| `nginx` | 80 / 443 | SSL termination |

> **Note:** The Ollama service has been removed. Internet access is required for Claude API calls. Embeddings remain fully local.

---

## Migrating from the Ollama version

1. Replace `.env` with the new `.env.example` (remove `LLM_PROVIDER`, `OLLAMA_*` vars, add `ANTHROPIC_API_KEY`)
2. Replace `docker-compose.yaml` (Ollama service and volume removed)
3. Replace `rag_pipeline.py`
4. Run `docker compose down && docker compose up -d`
5. Your existing pgvector index is compatible — click **Init** and start chatting

The `docs/` folder, `config.yaml`, `nginx.conf`, and `certs/` are unchanged.
