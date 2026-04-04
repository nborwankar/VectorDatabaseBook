"""
Chapter 7: Building a Scientific RAG System with PostgreSQL and pgvector
=========================================================================
A RAG system for scientific papers using pgvector for vector search
and Ollama for local LLM inference. Builds on Chapter 5's schema.

Dependencies: pip install psycopg2-binary sentence-transformers requests
              numpy PyMuPDF python-dotenv
Also requires: PostgreSQL with pgvector extension, Ollama running locally
"""

import os
import re
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
from sentence_transformers import SentenceTransformer
import requests

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'arxiv_papers'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'your_password'),
}

OLLAMA_BASE_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
DEFAULT_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.1:8b')


# =============================================================================
# 7.2 - Schema Setup (extends Chapter 5)
# =============================================================================

SCHEMA_SQL = """
-- Ensure pgvector is available
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Papers table (same as Ch5, included for standalone use)
CREATE TABLE IF NOT EXISTS papers (
    id SERIAL PRIMARY KEY,
    arxiv_id VARCHAR(50) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT,
    authors TEXT[],
    categories TEXT[],
    primary_category VARCHAR(50),
    published_date DATE,
    updated_date DATE,
    pdf_url TEXT,
    pdf_downloaded BOOLEAN DEFAULT FALSE,
    pdf_processed BOOLEAN DEFAULT FALSE,
    embedding_generated BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Paper chunks with embeddings
CREATE TABLE IF NOT EXISTS paper_chunks (
    id SERIAL PRIMARY KEY,
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_tokens INTEGER,
    embedding vector(384),
    section_name VARCHAR(255),
    page_number INTEGER,
    has_math BOOLEAN DEFAULT FALSE,
    has_code BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (paper_id, chunk_index)
);

-- HNSW index for fast vector search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON paper_chunks
USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_chunks_paper_id ON paper_chunks (paper_id);
CREATE INDEX IF NOT EXISTS idx_chunks_text_trgm ON paper_chunks
USING GIN (chunk_text gin_trgm_ops);

-- RAG-specific: conversation/query history
CREATE TABLE IF NOT EXISTS rag_conversations (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    context_chunks INTEGER[],
    model_used VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_rag_session ON rag_conversations (session_id, created_at);

-- Search analytics
CREATE TABLE IF NOT EXISTS search_analytics (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_embedding vector(384),
    num_results INTEGER,
    retrieval_time_ms INTEGER,
    generation_time_ms INTEGER,
    model_used VARCHAR(100),
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
    print("Database schema initialized.")


# =============================================================================
# 7.3 - Embedding Generator
# =============================================================================

class EmbeddingGenerator:
    """Singleton embedding generator for consistent embeddings."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        if self._initialized:
            return
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self._initialized = True
        print(f"Loaded {model_name} (dim={self.dimension})")

    def encode(self, texts, batch_size=32, show_progress=False):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(
            texts, batch_size=batch_size,
            show_progress_bar=show_progress, convert_to_numpy=True
        )

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a search query, returning a single vector."""
        return self.encode(query)[0]


# =============================================================================
# 7.4 - Text Chunking for Scientific Papers
# =============================================================================

class ScientificTextChunker:
    """Chunk scientific text preserving section boundaries and context."""

    SECTION_PATTERNS = [
        r'^(?:#{1,3})\s+(.+)',                          # Markdown headers
        r'^(\d+\.?\s+[A-Z][A-Za-z\s]+)',                 # Numbered sections
        r'^(Abstract|Introduction|Methods?|Results?|'
        r'Discussion|Conclusion|References|Appendix)',    # Named sections
    ]

    def __init__(self, target_size: int = 512, min_size: int = 128,
                 max_size: int = 1024, overlap: int = 64):
        self.target_size = target_size
        self.min_size = min_size
        self.max_size = max_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[Dict]:
        """Chunk text into overlapping segments with metadata."""
        sections = self._split_into_sections(text)
        all_chunks = []

        for section_name, section_text in sections:
            chunks = self._chunk_section(section_text, section_name)
            all_chunks.extend(chunks)

        # Assign sequential indices
        for i, chunk in enumerate(all_chunks):
            chunk['chunk_index'] = i

        return all_chunks

    def _split_into_sections(self, text: str) -> List[Tuple[str, str]]:
        """Split text into named sections."""
        lines = text.split('\n')
        sections = []
        current_section = "Introduction"
        current_lines = []

        for line in lines:
            is_header = False
            for pattern in self.SECTION_PATTERNS:
                match = re.match(pattern, line.strip(), re.IGNORECASE)
                if match:
                    if current_lines:
                        sections.append((current_section, '\n'.join(current_lines)))
                    current_section = match.group(1).strip() if match.group(1) else line.strip()
                    current_lines = []
                    is_header = True
                    break
            if not is_header:
                current_lines.append(line)

        if current_lines:
            sections.append((current_section, '\n'.join(current_lines)))

        return sections if sections else [("Full Text", text)]

    def _chunk_section(self, text: str, section_name: str) -> List[Dict]:
        """Chunk a single section with overlap."""
        words = text.split()
        if len(words) <= self.max_size:
            if len(words) >= self.min_size:
                return [{
                    'text': text.strip(),
                    'section': section_name,
                    'has_math': self._has_math(text),
                    'has_code': self._has_code(text),
                    'token_count': len(words)
                }]
            else:
                return []

        chunks = []
        start = 0

        while start < len(words):
            end = min(start + self.target_size, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)

            if len(chunk_words) >= self.min_size:
                chunks.append({
                    'text': chunk_text,
                    'section': section_name,
                    'has_math': self._has_math(chunk_text),
                    'has_code': self._has_code(chunk_text),
                    'token_count': len(chunk_words)
                })

            start += self.target_size - self.overlap

        return chunks

    @staticmethod
    def _has_math(text: str) -> bool:
        math_indicators = [r'\$.*\$', r'\\frac', r'\\sum', r'\\int',
                           r'\\alpha', r'\\beta', r'=\s*\d']
        return any(re.search(p, text) for p in math_indicators)

    @staticmethod
    def _has_code(text: str) -> bool:
        code_indicators = ['```', 'def ', 'import ', 'class ', 'print(',
                           'return ', 'if __name__']
        return any(ind in text for ind in code_indicators)


# =============================================================================
# 7.5 - Paper Ingestion Pipeline
# =============================================================================

class PaperIngestionPipeline:
    """Ingest papers: extract text, chunk, embed, store."""

    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.embedder = EmbeddingGenerator()
        self.chunker = ScientificTextChunker()

    def ingest_paper_text(self, arxiv_id: str, title: str, abstract: str,
                          full_text: str, authors: List[str] = None,
                          categories: List[str] = None,
                          published_date: str = None) -> Dict:
        """Ingest a paper from its text content."""
        conn = psycopg2.connect(**self.db_config)
        conn.autocommit = False

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Upsert paper record
                cur.execute("""
                    INSERT INTO papers (arxiv_id, title, abstract, authors,
                                        categories, primary_category, published_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (arxiv_id) DO UPDATE SET
                        title = EXCLUDED.title, abstract = EXCLUDED.abstract
                    RETURNING id
                """, (arxiv_id, title, abstract,
                      authors or [], categories or [],
                      categories[0] if categories else None,
                      published_date))
                paper_id = cur.fetchone()['id']

                # Chunk the text (abstract + full text)
                combined = f"Abstract\n{abstract}\n\n{full_text}"
                chunks = self.chunker.chunk_text(combined)

                if not chunks:
                    conn.rollback()
                    return {'paper_id': paper_id, 'chunks': 0, 'status': 'no_chunks'}

                # Generate embeddings
                texts = [c['text'] for c in chunks]
                embeddings = self.embedder.encode(texts, show_progress=True)

                # Delete existing chunks for this paper
                cur.execute("DELETE FROM paper_chunks WHERE paper_id = %s", (paper_id,))

                # Insert chunks with embeddings
                for chunk, embedding in zip(chunks, embeddings):
                    cur.execute("""
                        INSERT INTO paper_chunks
                        (paper_id, chunk_index, chunk_text, chunk_tokens,
                         embedding, section_name, has_math, has_code)
                        VALUES (%s, %s, %s, %s, %s::vector, %s, %s, %s)
                    """, (paper_id, chunk['chunk_index'], chunk['text'],
                          chunk['token_count'],
                          f"[{','.join(map(str, embedding.tolist()))}]",
                          chunk['section'], chunk['has_math'], chunk['has_code']))

                # Mark as processed
                cur.execute("""
                    UPDATE papers SET embedding_generated = TRUE WHERE id = %s
                """, (paper_id,))

                conn.commit()
                return {
                    'paper_id': paper_id,
                    'chunks': len(chunks),
                    'status': 'success'
                }

        except Exception as e:
            conn.rollback()
            logger.error(f"Ingestion error for {arxiv_id}: {e}")
            return {'paper_id': None, 'chunks': 0, 'status': f'error: {e}'}
        finally:
            conn.close()


# =============================================================================
# 7.6 - Semantic Search Engine
# =============================================================================

@dataclass
class SearchResult:
    """A single search result with paper metadata and matched chunks."""
    paper_id: int
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    published_date: Optional[str]
    matched_chunks: List[Dict] = field(default_factory=list)
    score: float = 0.0


class ScientificSearchEngine:
    """Semantic + hybrid search over scientific papers."""

    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.embedder = EmbeddingGenerator()

    def semantic_search(self, query: str, limit: int = 10,
                        categories: Optional[List[str]] = None,
                        min_year: Optional[int] = None) -> List[SearchResult]:
        """Pure vector similarity search."""
        query_embedding = self.embedder.encode_query(query)
        embedding_str = f"[{','.join(map(str, query_embedding.tolist()))}]"

        conn = psycopg2.connect(**self.db_config)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            sql = """
                SELECT
                    pc.id as chunk_id,
                    pc.chunk_text,
                    pc.section_name,
                    pc.chunk_index,
                    1 - (pc.embedding <=> %s::vector) as similarity,
                    p.id as paper_id,
                    p.arxiv_id,
                    p.title,
                    p.abstract,
                    p.authors,
                    p.categories,
                    p.published_date
                FROM paper_chunks pc
                JOIN papers p ON pc.paper_id = p.id
                WHERE 1=1
            """
            params = [embedding_str]

            if categories:
                sql += " AND p.categories && %s"
                params.append(categories)

            if min_year:
                sql += " AND EXTRACT(YEAR FROM p.published_date) >= %s"
                params.append(min_year)

            sql += " ORDER BY pc.embedding <=> %s::vector ASC LIMIT %s"
            params.extend([embedding_str, limit * 3])  # Over-fetch for grouping

            cur.execute(sql, params)
            rows = cur.fetchall()

        conn.close()
        return self._group_by_paper(rows, limit)

    def hybrid_search(self, query: str, limit: int = 10,
                      semantic_weight: float = 0.7,
                      categories: Optional[List[str]] = None) -> List[SearchResult]:
        """Hybrid search combining vector similarity and keyword matching."""
        query_embedding = self.embedder.encode_query(query)
        embedding_str = f"[{','.join(map(str, query_embedding.tolist()))}]"

        conn = psycopg2.connect(**self.db_config)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            sql = """
                WITH semantic AS (
                    SELECT pc.id as chunk_id,
                           pc.paper_id,
                           pc.chunk_text,
                           pc.section_name,
                           1 - (pc.embedding <=> %(emb)s::vector) as sem_score
                    FROM paper_chunks pc
                    JOIN papers p ON pc.paper_id = p.id
                    WHERE 1=1
                    {category_filter}
                    ORDER BY pc.embedding <=> %(emb)s::vector ASC
                    LIMIT %(fetch_limit)s
                ),
                keyword AS (
                    SELECT pc.id as chunk_id,
                           pc.paper_id,
                           pc.chunk_text,
                           pc.section_name,
                           ts_rank_cd(
                               to_tsvector('english', pc.chunk_text),
                               plainto_tsquery('english', %(query)s)
                           ) as kw_score
                    FROM paper_chunks pc
                    JOIN papers p ON pc.paper_id = p.id
                    WHERE to_tsvector('english', pc.chunk_text) @@
                          plainto_tsquery('english', %(query)s)
                    {category_filter}
                    LIMIT %(fetch_limit)s
                )
                SELECT
                    COALESCE(s.chunk_id, k.chunk_id) as chunk_id,
                    COALESCE(s.paper_id, k.paper_id) as paper_id,
                    COALESCE(s.chunk_text, k.chunk_text) as chunk_text,
                    COALESCE(s.section_name, k.section_name) as section_name,
                    COALESCE(s.sem_score, 0) as sem_score,
                    COALESCE(k.kw_score, 0) as kw_score,
                    (COALESCE(s.sem_score, 0) * %(sem_w)s +
                     COALESCE(k.kw_score, 0) * %(kw_w)s) as combined_score
                FROM semantic s
                FULL OUTER JOIN keyword k ON s.chunk_id = k.chunk_id
                ORDER BY combined_score DESC
                LIMIT %(fetch_limit)s
            """

            category_filter = ""
            if categories:
                category_filter = "AND p.categories && %(cats)s"

            sql = sql.format(category_filter=category_filter)

            params = {
                'emb': embedding_str,
                'query': query,
                'sem_w': semantic_weight,
                'kw_w': 1.0 - semantic_weight,
                'fetch_limit': limit * 3,
            }
            if categories:
                params['cats'] = categories

            cur.execute(sql, params)
            chunk_rows = cur.fetchall()

            # Fetch paper metadata for matched papers
            paper_ids = list(set(r['paper_id'] for r in chunk_rows if r['paper_id']))
            if not paper_ids:
                conn.close()
                return []

            cur.execute("""
                SELECT id, arxiv_id, title, abstract, authors,
                       categories, published_date
                FROM papers WHERE id = ANY(%s)
            """, (paper_ids,))
            papers_map = {r['id']: r for r in cur.fetchall()}

        conn.close()

        # Group chunks by paper
        results_map = {}
        for row in chunk_rows:
            pid = row['paper_id']
            if pid not in results_map and pid in papers_map:
                p = papers_map[pid]
                results_map[pid] = SearchResult(
                    paper_id=pid,
                    arxiv_id=p['arxiv_id'],
                    title=p['title'],
                    abstract=p['abstract'],
                    authors=p['authors'] or [],
                    categories=p['categories'] or [],
                    published_date=str(p['published_date']) if p['published_date'] else None,
                    matched_chunks=[],
                    score=0.0
                )
            if pid in results_map:
                results_map[pid].matched_chunks.append({
                    'text': row['chunk_text'],
                    'section': row['section_name'],
                    'score': float(row['combined_score'])
                })
                results_map[pid].score = max(
                    results_map[pid].score, float(row['combined_score'])
                )

        sorted_results = sorted(results_map.values(), key=lambda r: r.score, reverse=True)
        return sorted_results[:limit]

    def _group_by_paper(self, rows: List[Dict], limit: int) -> List[SearchResult]:
        """Group chunk-level results into paper-level results."""
        results_map = {}
        for row in rows:
            pid = row['paper_id']
            if pid not in results_map:
                results_map[pid] = SearchResult(
                    paper_id=pid,
                    arxiv_id=row['arxiv_id'],
                    title=row['title'],
                    abstract=row['abstract'],
                    authors=row['authors'] or [],
                    categories=row['categories'] or [],
                    published_date=str(row['published_date']) if row['published_date'] else None,
                    matched_chunks=[],
                    score=0.0
                )
            results_map[pid].matched_chunks.append({
                'text': row['chunk_text'],
                'section': row['section_name'],
                'score': float(row['similarity'])
            })
            results_map[pid].score = max(
                results_map[pid].score, float(row['similarity'])
            )

        sorted_results = sorted(results_map.values(), key=lambda r: r.score, reverse=True)
        return sorted_results[:limit]


# =============================================================================
# 7.7 - Ollama Integration
# =============================================================================

class OllamaClient:
    """Client for local Ollama LLM inference."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL,
                 default_model: str = DEFAULT_MODEL):
        self.base_url = base_url
        self.default_model = default_model

    def generate(self, prompt: str, model: str = None,
                 temperature: float = 0.1, max_tokens: int = 2048) -> str:
        """Generate a response from Ollama."""
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
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get('response', '')
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Start it with: ollama serve"
        except Exception as e:
            return f"Error: {str(e)}"

    def list_models(self) -> List[str]:
        """List available Ollama models."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return [m['name'] for m in r.json().get('models', [])]
        except Exception:
            return []

    def is_available(self) -> bool:
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=3)
            return True
        except Exception:
            return False


# =============================================================================
# 7.8 - RAG Pipeline
# =============================================================================

class ScientificRAG:
    """Complete RAG pipeline for scientific paper Q&A."""

    def __init__(self, db_config: dict, ollama_model: str = DEFAULT_MODEL):
        self.search_engine = ScientificSearchEngine(db_config)
        self.ollama = OllamaClient(default_model=ollama_model)
        self.db_config = db_config

    def answer(self, question: str, num_chunks: int = 5,
               search_mode: str = 'hybrid',
               categories: Optional[List[str]] = None) -> Dict:
        """Full RAG: retrieve relevant chunks, generate answer."""

        # 1. Retrieve
        t0 = time.time()
        if search_mode == 'hybrid':
            results = self.search_engine.hybrid_search(
                question, limit=num_chunks, categories=categories
            )
        else:
            results = self.search_engine.semantic_search(
                question, limit=num_chunks, categories=categories
            )
        retrieval_ms = (time.time() - t0) * 1000

        if not results:
            return {
                'answer': "No relevant papers found for your question.",
                'sources': [],
                'retrieval_ms': retrieval_ms,
                'generation_ms': 0
            }

        # 2. Build context
        context = self._build_context(results)

        # 3. Generate
        prompt = self._build_prompt(question, context)
        t1 = time.time()
        answer = self.ollama.generate(prompt)
        generation_ms = (time.time() - t1) * 1000

        # 4. Log analytics
        self._log_analytics(question, len(results), retrieval_ms, generation_ms)

        sources = [{
            'arxiv_id': r.arxiv_id,
            'title': r.title,
            'authors': r.authors[:3],
            'score': r.score
        } for r in results]

        return {
            'answer': answer,
            'sources': sources,
            'retrieval_ms': retrieval_ms,
            'generation_ms': generation_ms,
            'total_ms': retrieval_ms + generation_ms
        }

    def _build_context(self, results: List[SearchResult]) -> str:
        """Format search results as context for the LLM."""
        sections = []
        for i, result in enumerate(results, 1):
            chunks_text = "\n".join(
                c['text'] for c in result.matched_chunks[:3]
            )
            authors_str = ", ".join(result.authors[:3])
            if len(result.authors) > 3:
                authors_str += " et al."

            sections.append(f"""
[Source {i}] {result.title}
Authors: {authors_str}
ArXiv: {result.arxiv_id} | Published: {result.published_date or 'N/A'}
Relevance Score: {result.score:.3f}

{chunks_text}
""")
        return "\n---\n".join(sections)

    def _build_prompt(self, question: str, context: str) -> str:
        return f"""You are a scientific research assistant. Answer the question \
using ONLY the provided research paper excerpts.

Rules:
1. Base your answer strictly on the provided sources.
2. Cite sources using [Source N] notation.
3. If the sources don't contain enough information, say so clearly.
4. Use precise scientific language.
5. If sources disagree, note the disagreement.

RESEARCH PAPER EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""

    def _log_analytics(self, query: str, num_results: int,
                       retrieval_ms: float, generation_ms: float):
        """Log search analytics to database."""
        try:
            conn = psycopg2.connect(**self.db_config)
            with conn.cursor() as cur:
                embedding = self.search_engine.embedder.encode_query(query)
                embedding_str = f"[{','.join(map(str, embedding.tolist()))}]"
                cur.execute("""
                    INSERT INTO search_analytics
                    (query_text, query_embedding, num_results,
                     retrieval_time_ms, generation_time_ms, model_used)
                    VALUES (%s, %s::vector, %s, %s, %s, %s)
                """, (query, embedding_str, num_results,
                      int(retrieval_ms), int(generation_ms),
                      self.ollama.default_model))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to log analytics: {e}")


# =============================================================================
# 7.9 - Sample Data Loader
# =============================================================================

def load_sample_papers(db_config: dict):
    """Load sample papers for testing."""
    pipeline = PaperIngestionPipeline(db_config)

    sample_papers = [
        {
            'arxiv_id': '2301.00001',
            'title': 'Attention Is All You Need: A Retrospective',
            'abstract': 'The transformer architecture has revolutionized natural '
                        'language processing and beyond. This paper reviews the '
                        'impact of self-attention mechanisms on modern AI systems.',
            'full_text': """Introduction
The transformer architecture introduced self-attention as the primary mechanism
for sequence modeling. Unlike recurrent neural networks which process tokens
sequentially, transformers can attend to all positions simultaneously.

Methods
Self-attention computes queries, keys, and values from input embeddings.
The attention weights are computed as softmax(QK^T / sqrt(d_k))V. Multi-head
attention runs several attention functions in parallel, allowing the model
to attend to information from different representation subspaces.

Results
Transformers achieved state-of-the-art results on machine translation,
text summarization, and question answering benchmarks. The architecture
scales efficiently with compute and data, enabling models with billions
of parameters.

Conclusion
Self-attention has proven to be a versatile building block for neural
network architectures across NLP, computer vision, and scientific computing.""",
            'authors': ['A. Researcher', 'B. Scientist'],
            'categories': ['cs.CL', 'cs.AI'],
        },
        {
            'arxiv_id': '2301.00002',
            'title': 'Vector Databases for Large-Scale Similarity Search',
            'abstract': 'This paper surveys vector database technologies and their '
                        'applications in similarity search, recommendation systems, '
                        'and retrieval-augmented generation.',
            'full_text': """Introduction
Vector databases store high-dimensional embeddings and enable efficient
nearest neighbor search. They are essential infrastructure for modern
AI applications including semantic search and RAG systems.

Indexing Methods
Common indexing approaches include HNSW (Hierarchical Navigable Small
World graphs), IVF (Inverted File Index), and product quantization.
HNSW provides excellent recall with logarithmic search complexity.
IVF partitions the vector space into clusters for coarse-grained search.

Distance Metrics
Cosine similarity measures the angle between vectors and is widely used
for text embeddings. L2 (Euclidean) distance measures absolute distance
and is preferred for image embeddings. Inner product is useful when
vector magnitudes carry semantic meaning.

Applications
RAG systems use vector databases to retrieve relevant context for
language model generation. Recommendation engines find similar items.
Anomaly detection identifies outliers in embedding space.""",
            'authors': ['C. Engineer', 'D. Architect'],
            'categories': ['cs.DB', 'cs.IR'],
        }
    ]

    for paper in sample_papers:
        result = pipeline.ingest_paper_text(
            arxiv_id=paper['arxiv_id'],
            title=paper['title'],
            abstract=paper['abstract'],
            full_text=paper['full_text'],
            authors=paper['authors'],
            categories=paper['categories'],
            published_date='2023-01-01'
        )
        print(f"Ingested {paper['arxiv_id']}: {result}")


# =============================================================================
# 7.10 - Main
# =============================================================================

def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'setup':
            setup_database()
            return
        elif cmd == 'load-samples':
            load_sample_papers(DB_CONFIG)
            return

    print("=== Scientific RAG System ===\n")

    # Check Ollama
    ollama = OllamaClient()
    if not ollama.is_available():
        print("Ollama not running. Start with: ollama serve")
        print("Then pull a model: ollama pull llama3.1:8b")
        return

    models = ollama.list_models()
    print(f"Available Ollama models: {models}")

    # Initialize RAG
    rag = ScientificRAG(DB_CONFIG)

    # Demo questions
    demo_questions = [
        "How does self-attention work in transformers?",
        "What indexing methods are used in vector databases?",
        "Compare cosine similarity and L2 distance for embeddings."
    ]

    for q in demo_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print('-' * 60)
        result = rag.answer(q)
        print(f"A: {result['answer'][:500]}...")
        print(f"\nSources: {[s['arxiv_id'] for s in result['sources']]}")
        print(f"Retrieval: {result['retrieval_ms']:.0f}ms | "
              f"Generation: {result['generation_ms']:.0f}ms")

    # Interactive mode
    print(f"\n{'='*60}")
    print("Interactive mode (type 'quit' to exit)")
    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in ('quit', 'exit', 'q'):
            break
        if not q:
            continue
        result = rag.answer(q)
        print(f"\n{result['answer']}")
        print(f"\nSources: {[s['title'][:50] for s in result['sources']]}")

    print("Goodbye!")


if __name__ == "__main__":
    main()
