"""
Chapter 8: Building a Complete Conversation Search and RAG System
=================================================================
A production-oriented RAG system with:
- PostgreSQL + pgvector for vector storage
- Conversation history with session management
- FastAPI server with HTMX-compatible endpoints
- Ollama for local LLM inference
- Multi-turn conversation context

Dependencies: pip install fastapi uvicorn psycopg2-binary sentence-transformers
              requests numpy jinja2 python-multipart python-dotenv
Also requires: PostgreSQL with pgvector, Ollama running locally
"""

import os
import json
import time
import uuid
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
from sentence_transformers import SentenceTransformer
import requests

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'conversation_rag'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'your_password'),
}

OLLAMA_BASE_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
DEFAULT_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.1:8b')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')


# =============================================================================
# 8.2 - Database Schema
# =============================================================================

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Documents table (source material for RAG)
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(100) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT,
    doc_type VARCHAR(50) DEFAULT 'text',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document chunks with embeddings
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(384),
    section_name VARCHAR(255),
    token_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding ON document_chunks
USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_id ON document_chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_text_trgm ON document_chunks
USING GIN (chunk_text gin_trgm_ops);

-- Conversation sessions
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(64) UNIQUE NOT NULL,
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Conversation messages
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    sources JSONB DEFAULT '[]',
    retrieval_time_ms INTEGER,
    generation_time_ms INTEGER,
    model_used VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages (conversation_id, created_at);

-- Message embeddings for conversation search
CREATE TABLE IF NOT EXISTS message_embeddings (
    id SERIAL PRIMARY KEY,
    message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
    embedding vector(384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_msg_embeddings ON message_embeddings
USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Search history
CREATE TABLE IF NOT EXISTS search_history (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_embedding vector(384),
    search_type VARCHAR(20) DEFAULT 'hybrid',
    num_results INTEGER,
    execution_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def setup_database():
    """Initialize the database schema."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.close()
    print("Database schema created successfully.")


# =============================================================================
# 8.3 - Connection Pool
# =============================================================================

_connection_pool = None


def get_pool() -> pool.ThreadedConnectionPool:
    global _connection_pool
    if _connection_pool is None or _connection_pool.closed:
        _connection_pool = pool.ThreadedConnectionPool(
            minconn=2, maxconn=10, **DB_CONFIG
        )
    return _connection_pool


@contextmanager
def get_db_connection():
    """Context manager for database connections from pool."""
    p = get_pool()
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


# =============================================================================
# 8.4 - Embedding Generator
# =============================================================================

class EmbeddingGenerator:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        if self._initialized:
            return
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self._initialized = True
        print(f"Loaded {model_name} (dim={self.dimension})")

    def encode(self, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode(query)[0]

    def to_pg_vector(self, embedding: np.ndarray) -> str:
        return f"[{','.join(map(str, embedding.tolist()))}]"


# =============================================================================
# 8.5 - Document Ingestion
# =============================================================================

class DocumentIngester:
    """Ingest documents: chunk, embed, store."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.embedder = EmbeddingGenerator()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_document(self, doc_id: str, title: str, content: str,
                        source: str = None, doc_type: str = 'text',
                        metadata: dict = None) -> Dict:
        """Ingest a document: chunk, embed, store in pgvector."""
        chunks = self._chunk_text(content)

        if not chunks:
            return {'doc_id': doc_id, 'chunks': 0, 'status': 'no_content'}

        texts = [c['text'] for c in chunks]
        embeddings = self.embedder.encode(texts)

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Upsert document
                cur.execute("""
                    INSERT INTO documents (doc_id, title, content, source, doc_type, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        title = EXCLUDED.title, content = EXCLUDED.content
                    RETURNING id
                """, (doc_id, title, content, source, doc_type,
                      json.dumps(metadata or {})))
                document_id = cur.fetchone()['id']

                # Clear old chunks
                cur.execute("DELETE FROM document_chunks WHERE document_id = %s",
                            (document_id,))

                # Insert chunks with embeddings
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    cur.execute("""
                        INSERT INTO document_chunks
                        (document_id, chunk_index, chunk_text, embedding,
                         section_name, token_count)
                        VALUES (%s, %s, %s, %s::vector, %s, %s)
                    """, (document_id, i, chunk['text'],
                          self.embedder.to_pg_vector(emb),
                          chunk.get('section'), len(chunk['text'].split())))

        return {'doc_id': doc_id, 'chunks': len(chunks), 'status': 'success'}

    def _chunk_text(self, text: str) -> List[Dict]:
        """Simple word-based chunking with overlap."""
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = ' '.join(words[start:end])
            if len(chunk_text.split()) >= 20:  # Min chunk size
                chunks.append({'text': chunk_text, 'section': None})
            start += self.chunk_size - self.chunk_overlap

        return chunks


# =============================================================================
# 8.6 - Search Engine
# =============================================================================

@dataclass
class ChunkResult:
    chunk_id: int
    document_id: int
    doc_title: str
    chunk_text: str
    section: Optional[str]
    score: float
    doc_source: Optional[str] = None


class SearchEngine:
    """Hybrid search over document chunks."""

    def __init__(self):
        self.embedder = EmbeddingGenerator()

    def search(self, query: str, limit: int = 5,
               semantic_weight: float = 0.7) -> List[ChunkResult]:
        """Hybrid semantic + keyword search."""
        query_emb = self.embedder.encode_query(query)
        emb_str = self.embedder.to_pg_vector(query_emb)

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    WITH semantic AS (
                        SELECT dc.id as chunk_id,
                               dc.document_id,
                               dc.chunk_text,
                               dc.section_name,
                               1 - (dc.embedding <=> %(emb)s::vector) as sem_score
                        FROM document_chunks dc
                        ORDER BY dc.embedding <=> %(emb)s::vector ASC
                        LIMIT %(fetch)s
                    ),
                    keyword AS (
                        SELECT dc.id as chunk_id,
                               dc.document_id,
                               dc.chunk_text,
                               dc.section_name,
                               ts_rank_cd(
                                   to_tsvector('english', dc.chunk_text),
                                   plainto_tsquery('english', %(query)s)
                               ) as kw_score
                        FROM document_chunks dc
                        WHERE to_tsvector('english', dc.chunk_text) @@
                              plainto_tsquery('english', %(query)s)
                        LIMIT %(fetch)s
                    ),
                    combined AS (
                        SELECT
                            COALESCE(s.chunk_id, k.chunk_id) as chunk_id,
                            COALESCE(s.document_id, k.document_id) as document_id,
                            COALESCE(s.chunk_text, k.chunk_text) as chunk_text,
                            COALESCE(s.section_name, k.section_name) as section_name,
                            (COALESCE(s.sem_score, 0) * %(sw)s +
                             COALESCE(k.kw_score, 0) * %(kw)s) as score
                        FROM semantic s
                        FULL OUTER JOIN keyword k ON s.chunk_id = k.chunk_id
                    )
                    SELECT c.*, d.title as doc_title, d.source as doc_source
                    FROM combined c
                    JOIN documents d ON c.document_id = d.id
                    ORDER BY c.score DESC
                    LIMIT %(limit)s
                """, {
                    'emb': emb_str,
                    'query': query,
                    'sw': semantic_weight,
                    'kw': 1 - semantic_weight,
                    'fetch': limit * 3,
                    'limit': limit
                })
                rows = cur.fetchall()

        return [ChunkResult(
            chunk_id=r['chunk_id'],
            document_id=r['document_id'],
            doc_title=r['doc_title'],
            chunk_text=r['chunk_text'],
            section=r['section_name'],
            score=float(r['score']),
            doc_source=r.get('doc_source')
        ) for r in rows]

    def search_conversations(self, query: str, limit: int = 5) -> List[Dict]:
        """Search past conversation messages by semantic similarity."""
        query_emb = self.embedder.encode_query(query)
        emb_str = self.embedder.to_pg_vector(query_emb)

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT m.id, m.role, m.content, m.created_at,
                           c.session_id, c.title as conv_title,
                           1 - (me.embedding <=> %s::vector) as similarity
                    FROM message_embeddings me
                    JOIN messages m ON me.message_id = m.id
                    JOIN conversations c ON m.conversation_id = c.id
                    ORDER BY me.embedding <=> %s::vector ASC
                    LIMIT %s
                """, (emb_str, emb_str, limit))
                rows = cur.fetchall()

        return [dict(r) for r in rows]


# =============================================================================
# 8.7 - Conversation Manager
# =============================================================================

class ConversationManager:
    """Manage conversation sessions and message history."""

    def __init__(self):
        self.embedder = EmbeddingGenerator()

    def create_session(self, title: str = None) -> str:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())[:16]

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO conversations (session_id, title)
                    VALUES (%s, %s)
                """, (session_id, title or f"Conversation {session_id[:8]}"))

        return session_id

    def add_message(self, session_id: str, role: str, content: str,
                    sources: List[Dict] = None, retrieval_ms: int = None,
                    generation_ms: int = None, model: str = None) -> int:
        """Add a message to a conversation and embed it."""
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get conversation ID
                cur.execute("""
                    SELECT id FROM conversations WHERE session_id = %s
                """, (session_id,))
                row = cur.fetchone()
                if not row:
                    raise ValueError(f"Session {session_id} not found")
                conv_id = row['id']

                # Insert message
                cur.execute("""
                    INSERT INTO messages
                    (conversation_id, role, content, sources,
                     retrieval_time_ms, generation_time_ms, model_used)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (conv_id, role, content, json.dumps(sources or []),
                      retrieval_ms, generation_ms, model))
                msg_id = cur.fetchone()['id']

                # Embed and store message embedding
                embedding = self.embedder.encode_query(content)
                emb_str = self.embedder.to_pg_vector(embedding)
                cur.execute("""
                    INSERT INTO message_embeddings (message_id, embedding)
                    VALUES (%s, %s::vector)
                """, (msg_id, emb_str))

                # Update conversation timestamp
                cur.execute("""
                    UPDATE conversations SET updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (conv_id,))

        return msg_id

    def get_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        """Get conversation history for a session."""
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT m.role, m.content, m.sources, m.created_at
                    FROM messages m
                    JOIN conversations c ON m.conversation_id = c.id
                    WHERE c.session_id = %s
                    ORDER BY m.created_at ASC
                    LIMIT %s
                """, (session_id, limit))
                return [dict(r) for r in cur.fetchall()]

    def list_sessions(self, limit: int = 20) -> List[Dict]:
        """List recent conversation sessions."""
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT c.session_id, c.title, c.created_at, c.updated_at,
                           COUNT(m.id) as message_count
                    FROM conversations c
                    LEFT JOIN messages m ON c.id = m.conversation_id
                    GROUP BY c.id
                    ORDER BY c.updated_at DESC
                    LIMIT %s
                """, (limit,))
                return [dict(r) for r in cur.fetchall()]


# =============================================================================
# 8.8 - Ollama Client
# =============================================================================

class OllamaClient:
    """Client for Ollama LLM inference."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL,
                 default_model: str = DEFAULT_MODEL):
        self.base_url = base_url
        self.default_model = default_model

    def generate(self, prompt: str, model: str = None,
                 temperature: float = 0.1, max_tokens: int = 2048,
                 system: str = None) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model or self.default_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_ctx": 4096,
                "num_predict": max_tokens
            }
        }
        if system:
            payload["system"] = system

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get('response', '')
        except requests.exceptions.ConnectionError:
            return "Error: Ollama not available. Start with: ollama serve"
        except Exception as e:
            return f"Error: {str(e)}"

    def chat(self, messages: List[Dict], model: str = None,
             temperature: float = 0.1) -> str:
        """Multi-turn chat using Ollama's chat endpoint."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": 4096
            }
        }
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get('message', {}).get('content', '')
        except Exception as e:
            return f"Error: {str(e)}"

    def is_available(self) -> bool:
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=3)
            return True
        except Exception:
            return False

    def list_models(self) -> List[str]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return [m['name'] for m in r.json().get('models', [])]
        except Exception:
            return []


# =============================================================================
# 8.9 - RAG Engine
# =============================================================================

class ConversationRAG:
    """RAG engine with conversation context."""

    def __init__(self):
        self.search = SearchEngine()
        self.conversations = ConversationManager()
        self.ollama = OllamaClient()

    def ask(self, session_id: str, question: str,
            num_chunks: int = 5, include_history: bool = True) -> Dict:
        """Process a question with RAG and conversation context."""

        # 1. Retrieve relevant chunks
        t0 = time.time()
        chunks = self.search.search(question, limit=num_chunks)
        retrieval_ms = int((time.time() - t0) * 1000)

        # 2. Build context from chunks
        context = self._format_chunks(chunks)

        # 3. Get conversation history
        history = []
        if include_history:
            history = self.conversations.get_history(session_id, limit=10)

        # 4. Build prompt
        prompt = self._build_prompt(question, context, history)

        # 5. Generate answer
        t1 = time.time()
        answer = self.ollama.generate(prompt)
        generation_ms = int((time.time() - t1) * 1000)

        # 6. Store messages
        sources = [{
            'doc_title': c.doc_title,
            'chunk_text': c.chunk_text[:200],
            'score': c.score
        } for c in chunks]

        self.conversations.add_message(
            session_id, 'user', question
        )
        self.conversations.add_message(
            session_id, 'assistant', answer,
            sources=sources,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            model=self.ollama.default_model
        )

        return {
            'answer': answer,
            'sources': sources,
            'retrieval_ms': retrieval_ms,
            'generation_ms': generation_ms,
            'total_ms': retrieval_ms + generation_ms,
            'session_id': session_id
        }

    def _format_chunks(self, chunks: List[ChunkResult]) -> str:
        sections = []
        for i, chunk in enumerate(chunks, 1):
            sections.append(
                f"[Source {i}] From: {chunk.doc_title}\n"
                f"Score: {chunk.score:.3f}\n\n"
                f"{chunk.chunk_text}"
            )
        return "\n\n---\n\n".join(sections)

    def _build_prompt(self, question: str, context: str,
                      history: List[Dict]) -> str:
        history_text = ""
        if history:
            recent = history[-6:]  # Last 3 turns
            parts = []
            for msg in recent:
                role = "User" if msg['role'] == 'user' else "Assistant"
                parts.append(f"{role}: {msg['content'][:300]}")
            history_text = f"\nCONVERSATION HISTORY:\n" + "\n".join(parts) + "\n"

        return f"""You are a helpful research assistant. Answer the user's question \
using the provided sources and conversation history.

Rules:
1. Base your answer on the provided sources. Cite as [Source N].
2. If sources don't contain the answer, say so clearly.
3. Use the conversation history for context about follow-up questions.
4. Be concise but thorough.
{history_text}
RETRIEVED SOURCES:
{context}

QUESTION: {question}

ANSWER:"""

    def search_past_conversations(self, query: str, limit: int = 5) -> List[Dict]:
        """Search across past conversation messages."""
        return self.search.search_conversations(query, limit=limit)


# =============================================================================
# 8.10 - FastAPI Server
# =============================================================================

def create_app():
    """Create the FastAPI application."""
    from fastapi import FastAPI, Request, Form, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Conversation RAG System", version="1.0.0")

    rag = ConversationRAG()
    ingester = DocumentIngester()

    # --- API Endpoints ---

    @app.get("/api/health")
    def health_check():
        ollama_ok = rag.ollama.is_available()
        return {
            "status": "ok" if ollama_ok else "degraded",
            "ollama": ollama_ok,
            "models": rag.ollama.list_models() if ollama_ok else []
        }

    @app.post("/api/sessions")
    def create_session(title: str = None):
        session_id = rag.conversations.create_session(title)
        return {"session_id": session_id}

    @app.get("/api/sessions")
    def list_sessions(limit: int = 20):
        return rag.conversations.list_sessions(limit)

    @app.get("/api/sessions/{session_id}/messages")
    def get_messages(session_id: str, limit: int = 50):
        return rag.conversations.get_history(session_id, limit)

    @app.post("/api/ask")
    def ask_question(session_id: str, question: str):
        if not question.strip():
            raise HTTPException(400, "Question cannot be empty")
        result = rag.ask(session_id, question)
        return result

    @app.post("/api/documents")
    def ingest_document(doc_id: str, title: str, content: str,
                        source: str = None, doc_type: str = 'text'):
        result = ingester.ingest_document(doc_id, title, content, source, doc_type)
        return result

    @app.get("/api/search")
    def search_documents(q: str, limit: int = 5):
        chunks = rag.search.search(q, limit=limit)
        return [{
            'doc_title': c.doc_title,
            'chunk_text': c.chunk_text,
            'score': c.score,
            'section': c.section
        } for c in chunks]

    @app.get("/api/search/conversations")
    def search_conversations(q: str, limit: int = 5):
        return rag.search_past_conversations(q, limit)

    # --- HTMX-compatible HTML endpoints ---

    @app.post("/htmx/ask", response_class=HTMLResponse)
    async def htmx_ask(request: Request):
        form = await request.form()
        session_id = form.get('session_id', '')
        question = form.get('question', '')

        if not session_id:
            session_id = rag.conversations.create_session()

        result = rag.ask(session_id, question)

        sources_html = ""
        for i, src in enumerate(result['sources'], 1):
            sources_html += f"""
            <div class="source">
                <small>[Source {i}] {src['doc_title']} (score: {src['score']:.3f})</small>
            </div>"""

        return f"""
        <div class="message user-message">
            <strong>You:</strong> {question}
        </div>
        <div class="message assistant-message">
            <strong>Assistant:</strong>
            <p>{result['answer']}</p>
            <div class="sources">{sources_html}</div>
            <small class="timing">
                Retrieval: {result['retrieval_ms']}ms |
                Generation: {result['generation_ms']}ms
            </small>
        </div>
        <input type="hidden" name="session_id" value="{session_id}">
        """

    @app.get("/", response_class=HTMLResponse)
    def index():
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Conversation RAG</title>
            <script src="https://unpkg.com/htmx.org@1.9.10"></script>
            <style>
                body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .message { padding: 10px; margin: 10px 0; border-radius: 8px; }
                .user-message { background: #e3f2fd; }
                .assistant-message { background: #f5f5f5; }
                .sources { font-size: 0.85em; color: #666; margin-top: 8px; }
                .timing { color: #999; }
                #question-input { width: 70%; padding: 10px; }
                button { padding: 10px 20px; background: #1976d2; color: white;
                         border: none; border-radius: 4px; cursor: pointer; }
                button:hover { background: #1565c0; }
            </style>
        </head>
        <body>
            <h1>Conversation RAG System</h1>
            <div id="messages"></div>
            <form hx-post="/htmx/ask" hx-target="#messages" hx-swap="beforeend">
                <input type="hidden" name="session_id" id="session-id" value="">
                <input type="text" name="question" id="question-input"
                       placeholder="Ask a question..." autocomplete="off">
                <button type="submit">Ask</button>
            </form>
            <script>
                // Auto-create session on first load
                fetch('/api/sessions', {method: 'POST'})
                    .then(r => r.json())
                    .then(d => document.getElementById('session-id').value = d.session_id);
            </script>
        </body>
        </html>
        """

    return app


# =============================================================================
# 8.11 - Sample Data
# =============================================================================

def load_sample_data():
    """Load sample documents for testing."""
    ingester = DocumentIngester()

    docs = [
        {
            'doc_id': 'doc-transformers-101',
            'title': 'Understanding Transformer Architectures',
            'content': """Transformers are neural network architectures based on
self-attention mechanisms. Unlike RNNs that process sequences step by step,
transformers process all tokens in parallel. The key innovation is the
attention mechanism: Q (query), K (key), and V (value) matrices are computed
from input embeddings. Attention weights are softmax(QK^T/sqrt(d_k)) applied
to V. Multi-head attention runs multiple attention functions in parallel.

Transformers consist of encoder and decoder stacks. The encoder maps input
sequences to continuous representations. The decoder generates output tokens
autoregressively. Layer normalization and residual connections stabilize
training. Positional encodings inject sequence order information since
self-attention is permutation-invariant.

Modern variants include encoder-only models (BERT), decoder-only models
(GPT), and encoder-decoder models (T5). Scaling transformers to billions
of parameters has led to emergent capabilities in language understanding
and generation.""",
            'source': 'tutorial',
        },
        {
            'doc_id': 'doc-vector-db-guide',
            'title': 'Vector Database Implementation Guide',
            'content': """Vector databases are specialized systems for storing and
querying high-dimensional vector embeddings. They enable similarity search
where results are ranked by semantic closeness rather than exact keyword match.

Key indexing algorithms include HNSW (Hierarchical Navigable Small World),
which builds a multi-layer graph for efficient approximate nearest neighbor
search. IVF (Inverted File Index) partitions vectors into clusters, searching
only relevant partitions. Product quantization compresses vectors for memory
efficiency at the cost of some accuracy.

Distance metrics determine how similarity is measured. Cosine similarity
measures angle between vectors, ideal for normalized text embeddings.
Euclidean (L2) distance measures straight-line distance. Inner product
captures both direction and magnitude.

PostgreSQL with pgvector extension provides vector search within a
relational database. SQLite with the VSS extension offers a lightweight
embedded option. Both support HNSW indexing for fast approximate search.""",
            'source': 'guide',
        },
        {
            'doc_id': 'doc-rag-patterns',
            'title': 'RAG System Design Patterns',
            'content': """Retrieval-Augmented Generation (RAG) combines information
retrieval with language model generation. The basic pattern: embed a query,
search a vector store for relevant chunks, include them as context in a
prompt, and generate a grounded answer.

Advanced patterns include: Hybrid search combining semantic and keyword
retrieval for better recall. Re-ranking retrieved chunks with a
cross-encoder for better precision. Multi-step retrieval that refines
queries based on initial results. Conversation-aware RAG that uses
chat history to resolve coreferences and maintain context.

Chunking strategies matter. Fixed-size chunks are simple but may split
semantic units. Section-aware chunking preserves document structure.
Sliding windows with overlap ensure no information is lost at boundaries.
Hierarchical chunking stores both summaries and details.

Evaluation metrics include faithfulness (is the answer supported by
sources?), relevance (are the right chunks retrieved?), and correctness
(is the answer factually accurate?). Human evaluation remains the gold
standard for RAG quality assessment.""",
            'source': 'guide',
        }
    ]

    for doc in docs:
        result = ingester.ingest_document(**doc)
        print(f"Ingested '{doc['title']}': {result}")


# =============================================================================
# 8.12 - CLI Entry Point
# =============================================================================

def main():
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'setup':
            setup_database()
            return
        elif cmd == 'load-samples':
            load_sample_data()
            return
        elif cmd == 'serve':
            import uvicorn
            app = create_app()
            port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
            uvicorn.run(app, host="0.0.0.0", port=port)
            return

    print("Conversation RAG System")
    print("  python app.py setup         — Create database schema")
    print("  python app.py load-samples  — Load sample documents")
    print("  python app.py serve [port]  — Start FastAPI server")
    print()

    # Interactive CLI mode
    rag = ConversationRAG()

    if not rag.ollama.is_available():
        print("Ollama not running. Start with: ollama serve")
        return

    session_id = rag.conversations.create_session("CLI Session")
    print(f"Session: {session_id}")
    print("Type 'quit' to exit, 'search <query>' to search conversations\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ('quit', 'exit', 'q'):
            break
        if not q:
            continue
        if q.startswith('search '):
            query = q[7:]
            results = rag.search_past_conversations(query)
            for r in results:
                print(f"  [{r['role']}] {r['content'][:100]}... "
                      f"(sim: {r['similarity']:.3f})")
            continue

        result = rag.ask(session_id, q)
        print(f"\nAssistant: {result['answer']}")
        if result['sources']:
            print(f"  Sources: {[s['doc_title'] for s in result['sources']]}")
        print(f"  ({result['total_ms']}ms total)\n")


if __name__ == "__main__":
    main()
