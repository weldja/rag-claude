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

        last_message = [""]

        def progress_cb_with_capture(frac: float, msg: str):
            last_message[0] = msg
            progress_cb(frac, msg)

        def run():
            rebuild = (
                "incremental" if req.mode == "incremental"
                else True if req.mode == "full"
                else False
            )
            ok = rag.setup(rebuild=rebuild, progress_cb=progress_cb_with_capture)
            with lock:
                events.append({"done": True, "ok": ok, "message": last_message[0]})

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


# ── File serving ─────────────────────────────────────────────────────────────

from fastapi.responses import FileResponse
import urllib.parse

@app.get("/api/docs/{filename:path}")
def serve_doc(filename: str):
    """Serve a document file from the docs folder — searches recursively."""
    filename = urllib.parse.unquote(filename)
    docs_path = Path(_config.docs_path).resolve()

    # Try direct path first
    direct = (docs_path / filename).resolve()
    if str(direct).startswith(str(docs_path)) and direct.exists():
        return FileResponse(path=str(direct), filename=Path(filename).name)

    # Search recursively by filename only
    name_only = Path(filename).name
    for found in docs_path.rglob(name_only):
        if str(found.resolve()).startswith(str(docs_path)):
            return FileResponse(path=str(found), filename=name_only)

    raise HTTPException(404, f"File not found: {filename}")


# ── Diagnostics ──────────────────────────────────────────────────────────────

import io, zipfile, platform, time as _time

@app.get("/api/diagnostics")
def run_diagnostics():
    """Run all diagnostic checks and return results."""
    import subprocess, sys
    results = {}

    # 1. Database
    try:
        with rag._db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT version()")
            pg_ver = cur.fetchone()[0].split(",")[0]
            cur.execute("SELECT COUNT(*) FROM langchain_pg_embedding e JOIN langchain_pg_collection c ON e.collection_id=c.uuid WHERE c.name=%s", (_config.collection_name,))
            chunks = cur.fetchone()[0]
            cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
            db_size = cur.fetchone()[0]
        results["database"] = {"status": "ok", "version": pg_ver, "chunks": chunks, "size": db_size}
    except Exception as e:
        results["database"] = {"status": "error", "error": str(e)}

    # 2. API key
    key = os.getenv("ANTHROPIC_API_KEY", "") or load_saved_api_key()
    if key:
        results["api_key"] = {"status": "present", "prefix": key[:12] + "..."}
    else:
        results["api_key"] = {"status": "missing"}

    # 3. Anthropic connectivity
    try:
        import httpx
        t0 = _time.time()
        r = httpx.get("https://api.anthropic.com", timeout=5)
        ms = round((_time.time() - t0) * 1000)
        results["anthropic_connectivity"] = {"status": "ok", "response_ms": ms, "http_status": r.status_code}
    except Exception as e:
        results["anthropic_connectivity"] = {"status": "error", "error": str(e)}

    # 4. Docs folder
    docs_path = Path(_config.docs_path)
    folder_results = {"path": str(docs_path), "is_network_share": str(docs_path).startswith("\\") or str(docs_path).startswith("//"), "exists": docs_path.exists()}
    if docs_path.exists():
        files_status = []
        for f in docs_path.rglob("*"):
            if f.is_file() and not f.name.startswith("."):
                try:
                    size = f.stat().st_size
                    files_status.append({"name": f.name, "path": str(f.relative_to(docs_path)), "size_bytes": size, "readable": True})
                except Exception as e:
                    files_status.append({"name": f.name, "path": str(f), "readable": False, "error": str(e)})
        folder_results["files"] = files_status
        folder_results["total_files"] = len(files_status)
        folder_results["unreadable_files"] = sum(1 for f in files_status if not f["readable"])
        folder_results["status"] = "ok" if folder_results["unreadable_files"] == 0 else "warning"
    else:
        folder_results["status"] = "error"
        folder_results["error"] = "Docs folder does not exist"
    results["docs_folder"] = folder_results

    # 5. System info
    # Get git commit from env var baked in at build time
    git_commit = os.getenv("GIT_COMMIT", "unknown")

    results["system"] = {
        "weldai_version": "2.1.0",
        "git_commit": git_commit,
        "platform": platform.system(),
        "platform_version": platform.version()[:80],
        "python_version": sys.version.split()[0],
        "initialized": rag._initialized,
        "model": _config.claude_model,
        "embedding_model": _config.embedding_model,
    }

    # 6. Package versions
    try:
        import anthropic as _ant, langchain_core as _lc, fastembed as _fe
        results["packages"] = {
            "anthropic": _ant.__version__,
            "langchain_core": _lc.__version__,
            "fastembed": _fe.__version__,
        }
    except Exception:
        results["packages"] = {}

    # 7. Read backend log file
    try:
        log_path = "/app/weldai.log"
        if os.path.exists(log_path):
            with open(log_path) as lf:
                backend_logs = lf.readlines()[-100:]
            results["backend_logs"] = [l.rstrip() for l in backend_logs]
        else:
            results["backend_logs"] = ["Log file not yet created — restart the backend to enable logging"]
    except Exception as e:
        results["backend_logs"] = [f"Could not read log file: {str(e)}"]

    results["log_check"] = "ok"

    return results


@app.get("/api/diagnostics/download")
def download_diagnostics():
    """Generate and return a diagnostics zip file."""
    import json as _json

    diag = run_diagnostics()

    # Safe config (no API key)
    safe_cfg = {
        "docs_path": _config.docs_path,
        "db_host": _config.db_host,
        "db_port": _config.db_port,
        "db_name": _config.db_name,
        "collection_name": _config.collection_name,
        "embedding_model": _config.embedding_model,
        "claude_model": _config.claude_model,
        "chunk_size": _config.chunk_size,
        "chunk_overlap": _config.chunk_overlap,
        "retrieval_k": _config.retrieval_k,
    }

    # Stats
    stats = rag.get_stats() if rag._initialized else {}

    # Merge everything into a single diagnostics.json
    merged = {
        "generated": str(__import__("datetime").datetime.now()),
        "system": diag.get("system", {}),
        "database": diag.get("database", {}),
        "api_key": diag.get("api_key", {}),
        "anthropic_connectivity": diag.get("anthropic_connectivity", {}),
        "docs_folder": diag.get("docs_folder", {}),
        "packages": diag.get("packages", {}),
        "configuration": {
            "docs_path": safe_cfg.get("docs_path"),
            "collection_name": safe_cfg.get("collection_name"),
            "embedding_model": safe_cfg.get("embedding_model"),
            "claude_model": safe_cfg.get("claude_model"),
            "chunk_size": safe_cfg.get("chunk_size"),
            "chunk_overlap": safe_cfg.get("chunk_overlap"),
            "retrieval_k": safe_cfg.get("retrieval_k"),
            "db_host": safe_cfg.get("db_host"),
            "db_port": safe_cfg.get("db_port"),
        },
        "runtime": {
            "chunks_indexed": stats.get("chunks_indexed"),
            "documents_found": stats.get("documents_found"),
            "last_indexed": stats.get("last_indexed"),
            "cache_size": stats.get("cache_size"),
            "session_queries": stats.get("session_queries"),
            "session_tokens": stats.get("session_tokens"),
            "session_cost_usd": stats.get("session_cost_usd"),
        },
    }

    # Collect backend logs as separate file
    backend_logs = diag.get("backend_logs", [])
    backend_log_text = "\n".join(backend_logs) if isinstance(backend_logs, list) else str(backend_logs)

    # Build zip in memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("diagnostics.json", _json.dumps(merged, indent=2, default=str))
        zf.writestr("backend.log", backend_log_text)
        zf.writestr("readme.txt",
            "Weld AI Diagnostics Bundle\n"
            "Generated: " + str(__import__("datetime").datetime.now()) + "\n\n"
            "Contents:\n"
            "  diagnostics.json  - Complete system diagnostics (no API key included)\n"
            "  backend.log       - Recent backend application logs\n\n"
            "Please email this file to hello@weldai.uk for support.\n"
        )
    buf.seek(0)

    from fastapi.responses import StreamingResponse as SR
    ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    return SR(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=weldai_diagnostics_{ts}.zip"}
    )


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
