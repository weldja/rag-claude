"""
RAG Pipeline — Claude API Edition  (Production)
────────────────────────────────────────────────
Vectorstore : PostgreSQL + pgvector
Embeddings  : BAAI/bge-small-en-v1.5  via fastembed  (local, no API cost)
LLM         : Anthropic Claude API    (real streaming + token tracking)
"""

import contextlib
import os
import sys
import time
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import streamlit as st
import yaml
import anthropic
import psycopg2
import psycopg2.pool as pg_pool

# LangChain — document loading / chunking / vectorstore only
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Client config loader
# ─────────────────────────────────────────────

CONFIG_PATH = os.getenv("CLIENT_CONFIG", "config.yaml")

def load_client_config() -> dict:
    defaults = {
        "branding": {
            "company_name": "RAG Assistant",
            "tagline": "Ask questions across your documents",
            "accent_colour": "#185FA5",
            "assistant_name": "Document Assistant",
        },
        "suggested_questions": [
            "What does this document cover?",
            "Summarise the key points",
            "What are the main steps?",
        ],
        "ui": {
            "show_response_time": True,
            "show_sources": True,
            "show_copy_button": True,
            "max_suggested_questions": 5,
            "welcome_message": "Ask questions about your loaded documents.",
        },
    }
    try:
        if Path(CONFIG_PATH).exists():
            with open(CONFIG_PATH) as f:
                user = yaml.safe_load(f) or {}
            for section, values in user.items():
                if section in defaults and isinstance(values, dict):
                    defaults[section].update(values)
                else:
                    defaults[section] = values
    except Exception as e:
        logger.warning(f"Could not load {CONFIG_PATH}: {e}")
    return defaults


# ─────────────────────────────────────────────
# Claude pricing  ($ per million tokens)
# ─────────────────────────────────────────────

CLAUDE_PRICING: Dict[str, tuple] = {
    "claude-haiku-4-5-20251001": (0.80,   4.00),
    "claude-sonnet-4-6":         (3.00,  15.00),
    "claude-opus-4-6":          (15.00,  75.00),
}

def _model_rates(model: str):
    for key, (inp, out) in CLAUDE_PRICING.items():
        if model == key:
            return inp / 1_000_000, out / 1_000_000
    return 3.00 / 1_000_000, 15.00 / 1_000_000  # safe default


# ─────────────────────────────────────────────
# System configuration
# ─────────────────────────────────────────────

@dataclass
class Config:
    docs_path: str         = field(default_factory=lambda: os.getenv("DOCS_PATH", "docs"))
    db_host: str           = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    db_port: str           = field(default_factory=lambda: os.getenv("DB_PORT", "5432"))
    db_name: str           = field(default_factory=lambda: os.getenv("DB_NAME", "ragdb"))
    db_user: str           = field(default_factory=lambda: os.getenv("DB_USER", "rag"))
    db_password: str       = field(default_factory=lambda: os.getenv("DB_PASSWORD", "ragpass"))
    collection_name: str   = field(default_factory=lambda: os.getenv("COLLECTION_NAME", "rag_docs"))
    embedding_model: str   = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    chunk_size: int        = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")))
    chunk_overlap: int     = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100")))
    retrieval_k: int       = field(default_factory=lambda: int(os.getenv("RETRIEVAL_K", "4")))
    reranker_model: str    = field(default_factory=lambda: os.getenv("RERANKER_MODEL", ""))
    reranker_top_n: int    = field(default_factory=lambda: int(os.getenv("RERANKER_TOP_N", "4")))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    claude_model: str      = field(default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001"))
    max_tokens: int        = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "1024")))
    temperature: float     = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.1")))
    max_question_len: int  = field(default_factory=lambda: int(os.getenv("MAX_QUESTION_LEN", "2000")))
    batch_size: int        = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "100")))
    cache_ttl: int         = field(default_factory=lambda: int(os.getenv("CACHE_TTL", "86400")))

    @property
    def connection_string(self) -> str:
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    def validate(self) -> List[str]:
        """Return list of configuration errors (empty = valid)."""
        errors = []
        if not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is not set")
        elif not self.anthropic_api_key.startswith("sk-ant-"):
            errors.append("ANTHROPIC_API_KEY doesn't look valid (should start with sk-ant-)")
        if self.chunk_size < 100:
            errors.append("CHUNK_SIZE should be at least 100")
        if self.retrieval_k < 1:
            errors.append("RETRIEVAL_K should be at least 1")
        return errors


config = Config()


# ─────────────────────────────────────────────
# Cost / usage tracking
# ─────────────────────────────────────────────

@dataclass
class APIMetrics:
    total_queries: int = 0
    input_tokens: int  = 0
    output_tokens: int = 0
    model: str = field(default_factory=lambda: config.claude_model)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def total_cost(self) -> float:
        in_rate, out_rate = _model_rates(self.model)
        return self.input_tokens * in_rate + self.output_tokens * out_rate

    def add_usage(self, input_tokens: int, output_tokens: int):
        self.total_queries += 1
        self.input_tokens  += input_tokens
        self.output_tokens += output_tokens


# ─────────────────────────────────────────────
# Prompts — system/user separated for Claude
# ─────────────────────────────────────────────

# System prompt: stable instructions → good for prompt caching in future
SYSTEM_PROMPT = """\
You are a helpful AI assistant for business document search.
Answer questions using ONLY the context provided.
If the answer is not in the context, say exactly: \
"I don't have enough information to answer that."

Rules:
- Read ALL context chunks before answering
- Prioritise chunks that directly answer the question
- Be concise and specific
- Cite the source document name and page number where possible
- Do not make up or infer information not present in the context
- If multiple chunks give conflicting answers, use the most relevant one"""

# User prompt: context + question only
USER_PROMPT_TEMPLATE = """\
Context:
{context}

Question: {question}"""


# ─────────────────────────────────────────────
# Document loaders
# ─────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {
    ".txt", ".pdf", ".docx", ".pptx", ".xlsx", ".csv",
    ".doc", ".ppt", ".xls", ".rtf",
}


def load_document(filepath: Path) -> List[Document]:
    ext = filepath.suffix.lower()
    if ext in {".doc", ".ppt", ".xls", ".rtf"}:
        converted = _convert_legacy_office(filepath)
        if converted:
            return load_document(converted)
        logger.warning(f"Could not convert {filepath.name} — skipping")
        return []
    try:
        if ext == ".pdf":
            return PyPDFLoader(str(filepath)).load()
        elif ext == ".txt":
            return TextLoader(str(filepath), encoding="utf-8").load()
        elif ext == ".docx":
            return Docx2txtLoader(str(filepath)).load()
        elif ext == ".pptx":
            from pptx import Presentation
            prs = Presentation(str(filepath))
            text = [
                shape.text.strip()
                for slide in prs.slides
                for shape in slide.shapes
                if hasattr(shape, "text") and shape.text.strip()
            ]
            return [Document(page_content=" ".join(text), metadata={"source": str(filepath)})]
        elif ext in (".xlsx", ".csv"):
            import pandas as pd
            df = pd.read_excel(filepath) if ext == ".xlsx" else pd.read_csv(filepath)
            return [Document(page_content=df.to_string(index=False), metadata={"source": str(filepath)})]
        else:
            return []
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return []


def _convert_legacy_office(filepath: Path) -> Optional[Path]:
    import subprocess
    ext = filepath.suffix.lower()
    fmt = "docx" if ext in {".doc", ".rtf"} else ("pptx" if ext == ".ppt" else "xlsx")
    try:
        result = subprocess.run(
            ["libreoffice", "--headless", "--convert-to", fmt,
             "--outdir", str(filepath.parent), str(filepath)],
            capture_output=True, timeout=60,
        )
        if result.returncode == 0:
            converted = filepath.with_suffix("." + fmt)
            if converted.exists():
                return converted
    except Exception as e:
        logger.warning(f"LibreOffice conversion failed for {filepath}: {e}")
    return None


def collect_files(docs_path: str) -> List[Path]:
    root = Path(docs_path)
    if not root.exists():
        return []
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        for f in root.rglob(f"*{ext}"):
            if any(part.startswith(".") for part in f.parts[1:]):
                continue
            files.append(f)
    return sorted(files)


# ─────────────────────────────────────────────
# RAG System
# ─────────────────────────────────────────────

class RAGSystem:
    def __init__(self, cfg: Config = config):
        self.cfg = cfg
        self.embeddings: Optional[FastEmbedEmbeddings] = None
        self.vectorstore: Optional[PGVector] = None
        self.retriever = None
        self.reranker = None
        self._claude: Optional[anthropic.Anthropic] = None
        self._pool: Optional[pg_pool.SimpleConnectionPool] = None
        self.metrics = APIMetrics(model=cfg.claude_model)
        self._query_cache: Dict[str, Dict] = {}
        self.last_indexed: Optional[datetime] = None
        self._initialized = False

    # ── Embeddings (fastembed — no PyTorch) ───────────────────────────

    def init_embeddings(self):
        if self.embeddings is None:
            logger.info(f"Loading embedding model via fastembed: {self.cfg.embedding_model}")
            cache_dir = os.getenv("FASTEMBED_CACHE_PATH", None)
            self.embeddings = FastEmbedEmbeddings(
                model_name=self.cfg.embedding_model,
                **({"cache_dir": cache_dir} if cache_dir else {}),
            )

    # ── PostgreSQL connection pool ────────────────────────────────────

    def _init_pool(self):
        if self._pool is None:
            self._pool = pg_pool.SimpleConnectionPool(
                minconn=1,
                maxconn=5,
                host=self.cfg.db_host,
                port=self.cfg.db_port,
                dbname=self.cfg.db_name,
                user=self.cfg.db_user,
                password=self.cfg.db_password,
            )

    @contextlib.contextmanager
    def _db(self):
        """Yield a pooled database connection, return it when done."""
        self._init_pool()
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    # ── Vectorstore ───────────────────────────────────────────────────

    def connect_vectorstore(self):
        self.init_embeddings()
        self._init_pool()
        try:
            ts_file = Path(self.cfg.docs_path) / ".last_indexed"
            if ts_file.exists():
                self.last_indexed = datetime.fromisoformat(ts_file.read_text().strip())
        except Exception:
            pass
        self.vectorstore = PGVector(
            connection_string=self.cfg.connection_string,
            collection_name=self.cfg.collection_name,
            embedding_function=self.embeddings,
        )

    # ── QA setup (retriever + Claude client) ─────────────────────────

    def build_qa_chain(self):
        if self.vectorstore is None:
            raise RuntimeError("Vectorstore not ready.")

        errors = self.cfg.validate()
        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")

        # Anthropic client — max_retries handles transient network/overload errors
        self._claude = anthropic.Anthropic(
            api_key=self.cfg.anthropic_api_key,
            max_retries=3,
        )

        # Verify the key works (lightweight call, no cost)
        try:
            self._claude.models.list()
            logger.info(f"Claude API key verified. Model: {self.cfg.claude_model}")
        except anthropic.AuthenticationError:
            raise ValueError("ANTHROPIC_API_KEY is invalid. Check your .env file.")

        # Retriever — fetch more candidates when reranking
        retrieve_k = self.cfg.retrieval_k * 4 if self.cfg.reranker_model else self.cfg.retrieval_k
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retrieve_k},
        )

        # Optional cross-encoder reranker
        if self.cfg.reranker_model:
            try:
                from sentence_transformers import CrossEncoder
                model_path = Path("/app") / self.cfg.reranker_model
                if "models--" in str(model_path):
                    snapshots = model_path / "snapshots"
                    if snapshots.exists():
                        snap_dirs = list(snapshots.iterdir())
                        if snap_dirs:
                            model_path = snap_dirs[0]
                self.reranker = CrossEncoder(str(model_path))
                logger.info(f"Reranker loaded: {model_path}")
            except Exception as e:
                logger.error(f"Reranker failed to load: {e}")
                self.reranker = None

        self._initialized = True

    # ── Setup ─────────────────────────────────────────────────────────

    def setup(self, rebuild=False, progress_cb=None) -> bool:
        try:
            _cb(progress_cb, 0.05, "Loading embedding model...")
            self.init_embeddings()
            _cb(progress_cb, 0.15, "Connecting to vector store...")
            self.connect_vectorstore()
            if rebuild == "incremental":
                _cb(progress_cb, 0.20, "Checking for changes...")
                self.index_documents(progress_cb=progress_cb, progress_offset=0.20, incremental=True)
            elif rebuild:
                _cb(progress_cb, 0.20, "Re-indexing all documents...")
                self.index_documents(progress_cb=progress_cb, progress_offset=0.20, incremental=False)
            else:
                try:
                    test = self.vectorstore.similarity_search("test", k=1)
                    if not test:
                        _cb(progress_cb, 0.20, "No data found — indexing documents...")
                        self.index_documents(progress_cb=progress_cb, progress_offset=0.20)
                except Exception:
                    _cb(progress_cb, 0.20, "No data found — indexing documents...")
                    self.index_documents(progress_cb=progress_cb, progress_offset=0.20)
            _cb(progress_cb, 0.85, f"Connecting to Claude API ({self.cfg.claude_model})...")
            self.build_qa_chain()
            _cb(progress_cb, 1.00, "Ready!")
            return True
        except Exception as e:
            logger.exception("RAG setup failed")
            _cb(progress_cb, 1.00, f"Failed: {e}")
            return False

    # ── File hashing ──────────────────────────────────────────────────

    def _get_file_hash(self, filepath: Path) -> str:
        h = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _load_hash_store(self) -> Dict[str, str]:
        hash_file = Path(self.cfg.docs_path) / ".file_hashes.json"
        try:
            if hash_file.exists():
                return json.loads(hash_file.read_text())
        except Exception:
            pass
        return {}

    def _save_hash_store(self, hashes: Dict[str, str]):
        hash_file = Path(self.cfg.docs_path) / ".file_hashes.json"
        try:
            hash_file.write_text(json.dumps(hashes, indent=2))
        except Exception as e:
            logger.warning(f"Could not save hash store: {e}")

    # ── Document indexing ─────────────────────────────────────────────

    def index_documents(
        self,
        progress_cb=None,
        progress_offset: float = 0.0,
        incremental: bool = False,
    ) -> int:
        files = collect_files(self.cfg.docs_path)
        if not files:
            raise FileNotFoundError(f"No supported documents found in '{self.cfg.docs_path}'")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
            length_function=len,
        )

        if incremental:
            stored_hashes = self._load_hash_store()
            current_hashes: Dict[str, str] = {}
            new_or_changed: List[Path] = []

            for fp in files:
                fh = self._get_file_hash(fp)
                current_hashes[str(fp)] = fh
                if stored_hashes.get(str(fp)) != fh:
                    new_or_changed.append(fp)

            deleted = [k for k in stored_hashes if not Path(k).exists()]
            _cb(
                progress_cb, progress_offset + 0.05,
                f"Incremental: {len(new_or_changed)} changed, "
                f"{len(files) - len(new_or_changed)} unchanged, {len(deleted)} deleted",
            )

            if not new_or_changed and not deleted:
                _cb(progress_cb, progress_offset + 0.60, "All files up to date — nothing to index")
                self._persist_timestamp()
                return 0

            # Remove stale chunks from vectorstore using connection pool
            files_to_remove = [str(fp) for fp in new_or_changed] + deleted
            if files_to_remove:
                try:
                    with self._db() as conn:
                        cur = conn.cursor()
                        for source in files_to_remove:
                            cur.execute(
                                "DELETE FROM langchain_pg_embedding e "
                                "USING langchain_pg_collection c "
                                "WHERE e.collection_id = c.uuid "
                                "AND c.name = %s "
                                "AND e.cmetadata->>'source' = %s",
                                (self.cfg.collection_name, source),
                            )
                            logger.info(f"Removed chunks for: {source}")
                except Exception as e:
                    logger.warning(f"Could not remove old chunks: {e}")

            files_to_index = new_or_changed
            self._save_hash_store(current_hashes)
        else:
            files_to_index = files
            current_hashes = {str(fp): self._get_file_hash(fp) for fp in files}

        all_chunks: List[Document] = []
        for i, fp in enumerate(files_to_index):
            frac = (
                progress_offset + (0.60 * (i / len(files_to_index)))
                if files_to_index else progress_offset
            )
            _cb(progress_cb, frac, f"Loading {fp.name} ({i+1}/{len(files_to_index)})...")
            docs = load_document(fp)
            chunks = splitter.split_documents(docs)
            for chunk in chunks:
                source = Path(chunk.metadata.get("source", fp.name)).name
                page = chunk.metadata.get("page", "")
                page_str = f" | Page {int(page)+1}" if page != "" else ""
                chunk.page_content = f"[{source}{page_str}] " + chunk.page_content
            all_chunks.extend(chunks)
            logger.info(f"  {fp.name}: {len(docs)} pages → {len(chunks)} chunks")

        _cb(progress_cb, progress_offset + 0.60, f"Embedding {len(all_chunks)} chunks...")

        if not incremental:
            self.vectorstore = PGVector.from_documents(
                documents=all_chunks,
                embedding=self.embeddings,
                collection_name=self.cfg.collection_name,
                connection_string=self.cfg.connection_string,
                pre_delete_collection=True,
            )
            self._save_hash_store(current_hashes)
        else:
            if all_chunks:
                self.vectorstore.add_documents(all_chunks)

        self._persist_timestamp()
        logger.info(f"Indexed {len(all_chunks)} chunks from {len(files_to_index)} files")
        return len(all_chunks)

    def _persist_timestamp(self):
        self.last_indexed = datetime.now()
        try:
            with open("/app/docs/.last_indexed", "w") as f:
                f.write(self.last_indexed.isoformat())
        except Exception:
            pass

    # ── Query cache ───────────────────────────────────────────────────

    def _cache_key(self, question: str) -> str:
        return hashlib.md5(question.strip().lower().encode()).hexdigest()

    def _cache_get(self, question: str) -> Optional[Dict]:
        if self.cfg.cache_ttl <= 0:
            return None
        key = self._cache_key(question)
        entry = self._query_cache.get(key)
        if entry and (time.time() - entry["ts"]) < self.cfg.cache_ttl:
            return entry["result"]
        return None

    def _cache_set(self, question: str, result: Dict):
        if self.cfg.cache_ttl > 0:
            self._query_cache[self._cache_key(question)] = {"result": result, "ts": time.time()}

    # ── Input sanitisation ────────────────────────────────────────────

    def _sanitise(self, question: str) -> str:
        q = question.strip()
        if len(q) > self.cfg.max_question_len:
            q = q[: self.cfg.max_question_len]
            logger.warning("Question truncated to max length")
        return q

    # ── Retrieve + rerank (shared by ask and ask_streaming) ───────────

    def _retrieve(self, question: str) -> List[Document]:
        source_docs = self.retriever.invoke(question)
        if self.reranker and source_docs:
            pairs = [(question, doc.page_content) for doc in source_docs]
            scores = self.reranker.predict(pairs)
            scored = sorted(zip(scores, source_docs), key=lambda x: x[0], reverse=True)
            source_docs = [doc for _, doc in scored[: self.cfg.reranker_top_n]]
            logger.info(f"Reranked → {len(source_docs)} chunks")
        return source_docs

    def _build_sources(self, source_docs: List[Document]) -> List[Dict]:
        sources = []
        for doc in source_docs:
            page = doc.metadata.get("page", "")
            name = Path(doc.metadata.get("source", "unknown")).name
            sources.append({
                "content": doc.page_content[:300],
                "metadata": doc.metadata,
                "display_name": f"{name} p.{int(page)+1}" if page != "" else name,
            })
        return sources

    # ── Ask (non-streaming — used by CLI and cache reads) ─────────────

    def ask(self, question: str) -> Dict[str, Any]:
        if not self._initialized:
            return {"answer": "System not initialised. Click Init in the sidebar.",
                    "sources": [], "cached": False, "error": True}

        question = self._sanitise(question)
        cached = self._cache_get(question)
        if cached:
            logger.info("Cache hit")
            return {**cached, "cached": True}

        try:
            t0 = time.time()
            source_docs = self._retrieve(question)
            context = "\n\n".join(d.page_content for d in source_docs)

            response = self._claude.messages.create(
                model=self.cfg.claude_model,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                    context=context, question=question
                )}],
            )
            answer = response.content[0].text
            self.metrics.add_usage(response.usage.input_tokens, response.usage.output_tokens)
            logger.info(
                f"Claude — in:{response.usage.input_tokens} out:{response.usage.output_tokens} "
                f"session_cost:${self.metrics.total_cost:.5f}"
            )

            result = {
                "answer": answer,
                "sources": self._build_sources(source_docs),
                "elapsed": round(time.time() - t0, 1),
                "cached": False,
                "error": False,
            }
            self._cache_set(question, result)
            return result

        except anthropic.APIConnectionError:
            return {"answer": "⚠️ Could not reach the Anthropic API. Check your internet connection.",
                    "sources": [], "cached": False, "error": True}
        except anthropic.RateLimitError:
            return {"answer": "⚠️ Rate limit reached. Please wait a moment and try again.",
                    "sources": [], "cached": False, "error": True}
        except anthropic.AuthenticationError:
            return {"answer": "⚠️ Invalid API key. Check ANTHROPIC_API_KEY in your .env file.",
                    "sources": [], "cached": False, "error": True}
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {"answer": f"Error: {e}", "sources": [], "cached": False, "error": True}

    # ── Ask (streaming — used by the UI for live token output) ────────

    def ask_streaming(self, question: str, placeholder) -> Dict[str, Any]:
        """Stream Claude's response token-by-token into a Streamlit placeholder."""
        if not self._initialized:
            placeholder.error("System not initialised. Click Init in the sidebar.")
            return {"answer": "", "sources": [], "elapsed": 0, "cached": False, "error": True}

        question = self._sanitise(question)

        # Return cached result (fake-stream for consistency)
        cached = self._cache_get(question)
        if cached:
            logger.info("Cache hit")
            _stream_text(cached["answer"], placeholder)
            return {**cached, "cached": True}

        try:
            t0 = time.time()
            source_docs = self._retrieve(question)
            context = "\n\n".join(d.page_content for d in source_docs)

            buf = ""
            with self._claude.messages.stream(
                model=self.cfg.claude_model,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                    context=context, question=question
                )}],
            ) as stream:
                for text in stream.text_stream:
                    buf += text
                    placeholder.markdown(buf + "▌")
                placeholder.markdown(buf)

                final = stream.get_final_message()
                self.metrics.add_usage(
                    final.usage.input_tokens,
                    final.usage.output_tokens,
                )
                logger.info(
                    f"Claude stream — in:{final.usage.input_tokens} "
                    f"out:{final.usage.output_tokens} "
                    f"session_cost:${self.metrics.total_cost:.5f}"
                )

            result = {
                "answer": buf,
                "sources": self._build_sources(source_docs),
                "elapsed": round(time.time() - t0, 1),
                "cached": False,
                "error": False,
            }
            self._cache_set(question, result)
            return result

        except anthropic.APIConnectionError as e:
            msg = "⚠️ Could not reach the Anthropic API. Check your internet connection."
            placeholder.error(msg)
            return {"answer": msg, "sources": [], "elapsed": 0, "cached": False, "error": True}
        except anthropic.RateLimitError:
            msg = "⚠️ Rate limit reached. Please wait a moment and try again."
            placeholder.error(msg)
            return {"answer": msg, "sources": [], "elapsed": 0, "cached": False, "error": True}
        except anthropic.AuthenticationError:
            msg = "⚠️ Invalid API key. Check ANTHROPIC_API_KEY in your .env file."
            placeholder.error(msg)
            return {"answer": msg, "sources": [], "elapsed": 0, "cached": False, "error": True}
        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            placeholder.error(f"Error: {e}")
            return {"answer": str(e), "sources": [], "elapsed": 0, "cached": False, "error": True}

    # ── Stats ─────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        count = 0
        if self.vectorstore:
            try:
                with self._db() as conn:
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT COUNT(*) FROM langchain_pg_embedding e "
                        "JOIN langchain_pg_collection c ON e.collection_id = c.uuid "
                        "WHERE c.name = %s",
                        (self.cfg.collection_name,),
                    )
                    count = cur.fetchone()[0]
            except Exception:
                count = -1
        files = collect_files(self.cfg.docs_path)
        m = self.metrics
        return {
            "chunks_indexed": count,
            "documents_found": len(files),
            "embedding_model": self.cfg.embedding_model,
            "llm_provider": "claude",
            "llm_model": self.cfg.claude_model,
            "chunk_size": self.cfg.chunk_size,
            "retrieval_k": self.cfg.retrieval_k,
            "cache_size": len(self._query_cache),
            "last_indexed": (
                self.last_indexed.strftime("%Y-%m-%d %H:%M") if self.last_indexed else "Never"
            ),
            "session_queries": m.total_queries,
            "session_tokens": m.total_tokens,
            "session_cost_usd": round(m.total_cost, 5),
        }


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _cb(cb, progress: float, message: str):
    if cb:
        cb(progress, message)
    logger.info(f"[{progress:.0%}] {message}")


def _stream_text(text: str, placeholder, delay: float = 0.012):
    """Fake word-by-word stream for cached responses."""
    words = text.split()
    buf = ""
    for word in words:
        buf += word + " "
        placeholder.markdown(buf + "▌")
        time.sleep(delay)
    placeholder.markdown(buf.strip())


@st.cache_resource(show_spinner=False)
def get_rag_system() -> RAGSystem:
    return RAGSystem()


@st.cache_data(show_spinner=False)
def get_client_config() -> dict:
    return load_client_config()


def _handle_question(rag: RAGSystem, question: str, cfg: dict):
    """Process a question and append to session-state messages."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        result = rag.ask_streaming(question, placeholder)   # real streaming
        if not result["error"]:
            ui_cfg = cfg.get("ui", {})
            if ui_cfg.get("show_sources", True) and result["sources"]:
                _render_sources(result["sources"])
            meta = []
            if result.get("cached"):
                meta.append("⚡ cached")
            if ui_cfg.get("show_response_time", True) and result.get("elapsed"):
                meta.append(f"⏱ {result['elapsed']}s")
            if meta:
                st.caption(" · ".join(meta))
            if ui_cfg.get("show_copy_button", True) and result["answer"]:
                st.button(
                    "📋 Copy answer",
                    key=f"copy_{hashlib.md5(question.encode()).hexdigest()[:8]}",
                    on_click=lambda: st.write(
                        f'<script>navigator.clipboard.writeText({json.dumps(result["answer"])})</script>',
                        unsafe_allow_html=True,
                    ),
                )
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
        "cached": result.get("cached", False),
        "elapsed": result.get("elapsed"),
    })


def _render_sources(sources: List[Dict]):
    if not sources:
        return
    with st.expander(f"📚 {len(sources)} source(s)"):
        for i, src in enumerate(sources, 1):
            label = src.get("display_name") or Path(
                src["metadata"].get("source", "unknown")
            ).name
            st.markdown(f"**Source {i}** — `{label}`")
            st.text(src["content"])
            st.markdown("---")


# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────

def run_ui():
    cfg            = get_client_config()
    branding       = cfg.get("branding", {})
    ui_cfg         = cfg.get("ui", {})
    suggested      = cfg.get("suggested_questions", [])
    accent         = branding.get("accent_colour", "#185FA5")
    company        = branding.get("company_name", "RAG Assistant")
    assistant_name = branding.get("assistant_name", "Document Assistant")
    tagline        = branding.get("tagline", "Ask questions across your documents")

    st.set_page_config(
        page_title=f"{company} — {assistant_name}",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(f"""
    <style>
    .brand-header {{
        background: {accent}18;
        border-left: 4px solid {accent};
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-bottom: 1rem;
    }}
    .brand-header h1 {{ color: {accent}; font-size: 1.4rem; margin: 0 0 4px 0; }}
    .brand-header p  {{ color: #5F5E5A;  font-size: 0.9rem; margin: 0; }}
    .cost-pill {{
        background: {accent}12;
        border: 1px solid {accent}40;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.78rem;
        color: {accent};
        font-weight: 600;
        display: inline-block;
        margin-top: 4px;
    }}
    </style>
    """, unsafe_allow_html=True)

    rag = get_rag_system()

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div class="brand-header">
            <h1>{company}</h1>
            <p>{assistant_name}</p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("⚙️ Configuration"):
            st.caption(f"**LLM:** Claude API")
            st.caption(f"**Model:** `{config.claude_model}`")
            st.caption(f"**Embeddings:** `{config.embedding_model}` (local)")
            if not config.anthropic_api_key:
                st.warning("⚠️ ANTHROPIC_API_KEY not set")
            elif rag._initialized:
                m = rag.metrics
                st.markdown(
                    f'<span class="cost-pill">'
                    f'💰 ${m.total_cost:.4f} · {m.total_tokens:,} tokens · {m.total_queries} queries'
                    f'</span>',
                    unsafe_allow_html=True,
                )
                if m.total_queries:
                    st.caption(f"Avg cost/query: ${m.total_cost / m.total_queries:.5f}")

        st.markdown("---")

        st.markdown("**📂 Documents**")
        files = collect_files(config.docs_path)
        if files:
            st.success(f"{len(files)} file(s) loaded")
            with st.expander("View files"):
                for f in files:
                    st.text(f"📄 {f.name}")
        else:
            st.warning("No documents found in docs/")

        with st.expander("ℹ️ When to use each button"):
            st.markdown("""
**🚀 Init** — Every visit. Connects to existing index, no re-indexing.

**⚡ Smart** — After adding/changing documents. Processes only new/changed files.

**🔄 Full** — First time setup or full re-index from scratch.
            """)

        col1, col2, col3 = st.columns(3)
        with col1:
            init_btn    = st.button("🚀 Init",  use_container_width=True)
        with col2:
            update_btn  = st.button("⚡ Smart", use_container_width=True)
        with col3:
            rebuild_btn = st.button("🔄 Full",  use_container_width=True)

        ts_file = Path("/app/docs") / ".last_indexed"
        if ts_file.exists():
            try:
                last_ts = datetime.fromisoformat(ts_file.read_text().strip())
                new_files = [f for f in files if f.stat().st_mtime > last_ts.timestamp()]
                if new_files:
                    st.warning(f"⚠️ {len(new_files)} new/changed file(s) — click Smart")
            except Exception:
                pass
        elif files:
            st.warning("⚠️ Documents not yet indexed — click Smart or Full")

        if rag._initialized:
            st.markdown("---")
            st.markdown("**📊 Stats**")
            stats = rag.get_stats()
            c1, c2 = st.columns(2)
            c1.metric("Chunks", stats["chunks_indexed"])
            c2.metric("Cache",  stats["cache_size"])
            st.caption(f"Last indexed: {stats['last_indexed']}")
            if st.button("🗑️ Clear cache", use_container_width=True):
                rag._query_cache.clear()
                st.success("Cache cleared")

        if rag._initialized and suggested:
            st.markdown("---")
            st.markdown("**💡 Suggested questions**")
            max_q = ui_cfg.get("max_suggested_questions", 5)
            for i, q in enumerate(suggested[:max_q]):
                if st.button(q, key=f"sq_{i}", use_container_width=True):
                    st.session_state["pending_question"] = q
                    st.rerun()

    # ── Init / Rebuild ────────────────────────────────────────────────
    if init_btn or rebuild_btn or update_btn:
        progress_bar = st.progress(0.0)
        status_txt   = st.empty()

        def on_progress(frac: float, msg: str):
            progress_bar.progress(min(frac, 1.0))
            status_txt.text(msg)

        rebuild_mode = "incremental" if update_btn else rebuild_btn
        ok = rag.setup(rebuild=rebuild_mode, progress_cb=on_progress)
        time.sleep(0.3)
        progress_bar.empty()
        status_txt.empty()
        if ok:
            st.success("✅ Ready!")
            st.rerun()
        else:
            st.error("❌ Initialisation failed. Check logs.")

    # ── Not yet initialised ───────────────────────────────────────────
    if not rag._initialized:
        st.markdown(f"""
        <div class="brand-header">
            <h1>{company}</h1>
            <p>{tagline}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(ui_cfg.get("welcome_message", "Ask questions about your loaded documents."))
        if not config.anthropic_api_key:
            st.error(
                "**ANTHROPIC_API_KEY is not configured.**\n\n"
                "Add it to your `.env` file:\n```\nANTHROPIC_API_KEY=sk-ant-...\n```\n"
                "Then restart the container."
            )
        st.info("""
| Button | When to use |
|--------|-------------|
| 🚀 **Init** | Every visit — connects to your existing index |
| ⚡ **Smart** | After adding or changing documents |
| 🔄 **Full** | First time setup, or full reset |
        """)
        st.markdown("**Supported:** PDF · DOCX · PPTX · TXT · XLSX · CSV · DOC · PPT · XLS · RTF")
        return

    tab_chat, tab_search, tab_about = st.tabs(["💬 Chat", "🔎 Search", "ℹ️ About"])

    # ── Chat ─────────────────────────────────────────────────────────
    with tab_chat:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        pending = st.session_state.pop("pending_question", None)

        if prompt := st.chat_input(f"Ask {assistant_name} a question..."):
            _handle_question(rag, prompt, cfg)
            st.rerun()

        for msg in reversed(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    if ui_cfg.get("show_sources", True) and msg.get("sources"):
                        _render_sources(msg["sources"])
                    meta = []
                    if msg.get("cached"):
                        meta.append("⚡ cached")
                    if ui_cfg.get("show_response_time", True) and msg.get("elapsed"):
                        meta.append(f"⏱ {msg['elapsed']}s")
                    if meta:
                        st.caption(" · ".join(meta))

        if pending:
            _handle_question(rag, pending, cfg)
            st.rerun()

        if st.session_state.messages:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear conversation"):
                    st.session_state.messages = []
                    st.rerun()
            with col2:
                last_user = next(
                    (m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"),
                    None,
                )
                if last_user and st.button("🔁 Ask again"):
                    _handle_question(rag, last_user, cfg)
                    st.rerun()

    # ── Search ────────────────────────────────────────────────────────
    with tab_search:
        st.subheader("Search document chunks directly (no LLM)")
        query = st.text_input("Search query")
        k = st.slider("Number of results", 1, 10, 4)
        if st.button("🔎 Search") and query:
            with st.spinner("Searching..."):
                docs = rag.vectorstore.similarity_search(query, k=k)
            for i, doc in enumerate(docs, 1):
                src   = Path(doc.metadata.get("source", "unknown")).name
                page  = doc.metadata.get("page", "")
                label = f"{src} p.{int(page)+1}" if page != "" else src
                with st.expander(f"Result {i} — {label}"):
                    st.text(doc.page_content)

    # ── About ─────────────────────────────────────────────────────────
    with tab_about:
        st.subheader(f"About {company} — {assistant_name}")
        st.markdown(f"""
**LLM:** Claude API (`{config.claude_model}`)
**Embeddings:** `{config.embedding_model}` (local via fastembed)
**Supported formats:** PDF · DOCX · PPTX · TXT · XLSX · CSV
        """)
        st.json(rag.get_stats())


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def cli_index():
    print(f"[{datetime.now()}] Indexing '{config.docs_path}'...")
    rag = RAGSystem()
    rag.init_embeddings()
    rag.connect_vectorstore()
    n = rag.index_documents()
    print(f"[{datetime.now()}] Done — {n} chunks indexed.")

def cli_query(question: str):
    rag = RAGSystem()
    rag.setup()
    result = rag.ask(question)
    print(f"\nAnswer:\n{result['answer']}\n")
    for i, src in enumerate(result["sources"], 1):
        print(f"Source {i}: {src.get('display_name', '?')}")

def cli_status():
    rag = RAGSystem()
    rag.init_embeddings()
    rag.connect_vectorstore()
    print(json.dumps(rag.get_stats(), indent=2))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG Pipeline — Claude Edition")
    parser.add_argument("--mode", choices=["ui", "index", "query", "status"], default="ui")
    parser.add_argument("--query", type=str, default="")
    args = parser.parse_args()
    if args.mode == "index":
        cli_index()
    elif args.mode == "query":
        if not args.query:
            print("--query is required"); sys.exit(1)
        cli_query(args.query)
    elif args.mode == "status":
        cli_status()
    else:
        run_ui()
