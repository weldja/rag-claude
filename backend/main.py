"""
FastAPI backend for Weld AI RAG
"""

import json
import logging
import os
import threading
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pathlib import Path
from rag_core import (
    RAGSystem, Config, load_client_config, load_saved_api_key,
    save_api_key, delete_api_key, collect_files, file_icon,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Weld AI RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global RAG instance (thread-safe via internal lock)
_config = Config()
rag = RAGSystem(_config)


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    session_id: str = "default"
    history: list = []  # [{role, content}] last N turns

class SetupRequest(BaseModel):
    mode: str = "init"   # "init" | "full" | "incremental"

class ApiKeyRequest(BaseModel):
    key: str


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def get_config():
    return load_client_config()


@app.get("/api/status")
def get_status():
    files = collect_files(_config.docs_path)
    chunks = rag.get_chunks_in_db()
    has_key = bool(
        os.getenv("ANTHROPIC_API_KEY", "") or load_saved_api_key()
    )
    changes = {}
    if chunks > 0:
        changes = rag.get_changes()

    return {
        "initialized": rag._initialized,
        "chunks_in_db": chunks,
        "has_key": has_key,
        "key_source": "env" if os.getenv("ANTHROPIC_API_KEY") else ("saved" if load_saved_api_key() else "none"),
        "model": _config.claude_model,
        "files": [
            {
                "name": f.name,
                "ext": f.suffix.lower().lstrip("."),
                "icon": file_icon(f.name),
                "size": f.stat().st_size,
            }
            for f in files
        ],
        "changes": changes,
        "stats": rag.get_stats() if rag._initialized else None,
    }


@app.post("/api/apikey")
def set_api_key(req: ApiKeyRequest):
    if not req.key.startswith("sk-ant-"):
        raise HTTPException(400, "Key must start with sk-ant-")
    save_api_key(req.key)
    _config.anthropic_api_key = req.key
    return {"ok": True}


@app.delete("/api/apikey")
def remove_api_key():
    delete_api_key()
    _config.anthropic_api_key = ""
    rag._initialized = False
    rag._claude = None
    return {"ok": True}


@app.post("/api/setup")
def setup(req: SetupRequest):
    """SSE stream of setup progress."""

    def event_stream():
        events = []
        lock = threading.Lock()

        def progress_cb(frac: float, msg: str):
            with lock:
                events.append({"progress": round(frac, 3), "message": msg})

        def run():
            rebuild = (
                "incremental" if req.mode == "incremental"
                else True if req.mode == "full"
                else False
            )
            ok = rag.setup(rebuild=rebuild, progress_cb=progress_cb)
            with lock:
                events.append({"done": True, "ok": ok})

        t = threading.Thread(target=run, daemon=True)
        t.start()

        import time
        while t.is_alive() or events:
            with lock:
                batch = events[:]
                events.clear()
            for ev in batch:
                yield f"data: {json.dumps(ev)}\n\n"
            if not batch:
                time.sleep(0.1)

        # Drain any remaining events
        with lock:
            for ev in events:
                yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/ask")
def ask(req: AskRequest):
    """SSE stream of answer tokens. Saves Q&A to history."""

    def event_stream():
        # Save user message first
        rag.save_message(req.session_id, "user", req.question)

        answer_buf = ""
        sources_buf = []
        elapsed_buf = None
        cached_buf = False

        for event in rag.ask_stream(req.question, history=req.history):
            yield f"data: {json.dumps(event)}\n\n"
            # Collect answer for saving
            if event["type"] == "token":
                answer_buf += event["data"]
            elif event["type"] == "cached":
                answer_buf = event["data"]
                cached_buf = True
            elif event["type"] == "sources":
                sources_buf = event["data"]
            elif event["type"] == "meta":
                elapsed_buf = event["data"].get("elapsed")
            elif event["type"] == "done" and answer_buf:
                rag.save_message(
                    req.session_id, "assistant", answer_buf,
                    sources=sources_buf, elapsed=elapsed_buf, cached=cached_buf
                )

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.delete("/api/cache")
def clear_cache():
    rag.clear_cache()
    return {"ok": True}


# ── Chat history endpoints ────────────────────────────────────────────────────

@app.get("/api/history/{session_id}")
def get_history(session_id: str, limit: int = 50):
    return {"messages": rag.get_history(session_id, limit=limit)}


@app.delete("/api/history/{session_id}")
def clear_history(session_id: str):
    rag.clear_history(session_id)
    return {"ok": True}


@app.get("/api/sessions")
def list_sessions():
    return {"sessions": rag.list_sessions()}


@app.on_event("startup")
def startup():
    """Ensure DB tables exist on startup."""
    import time
    # Wait briefly for DB to be ready
    for _ in range(5):
        try:
            rag._init_pool()
            rag.ensure_history_table()
            logger.info("Chat history table ready")
            break
        except Exception as e:
            logger.warning(f"DB not ready yet: {e}")
            time.sleep(2)


class SearchRequest(BaseModel):
    query: str
    k: int = 6

@app.post("/api/search")
def search(req: SearchRequest):
    if not rag._initialized or not rag.vectorstore:
        raise HTTPException(400, "System not initialised")
    docs = rag.vectorstore.similarity_search(req.query, k=req.k)
    results = []
    for doc in docs:
        src = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page", "")
        results.append({
            "display_name": f"{src} p.{int(page)+1}" if page != "" else src,
            "content": doc.page_content[:400],
            "icon": file_icon(src),
        })
    return {"results": results}
