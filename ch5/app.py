"""
Chapter 5: Building an ArXiv Paper Search System with PostgreSQL pgvector
=========================================================================
Architecture scaffold with class/method definitions.

NOTE FROM THE BOOK: "This whole chapter outlines an architecture via class
and method definitions. Code implementing these is on the accompanying
github repository mentioned in the preface."

Methods marked with `pass` are STUBS — implement them or pull from the
companion repo. SQL schema and some key methods (_upsert_paper, Singleton
EmbeddingGenerator) are fully provided.

Dependencies: pip install arxiv PyMuPDF psycopg2-binary sentence-transformers
              requests numpy tqdm python-dotenv
"""

import time
import logging
import os
import hashlib
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Generator, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from urllib.parse import quote

import arxiv
import requests
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# =============================================================================
# Section 3 — Environment & Config
# =============================================================================

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'arxiv_papers'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'your_password'),
}


# =============================================================================
# Section 4 — Database Schema (COMPLETE SQL)
# =============================================================================

SCHEMA_SQL = """
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Main papers table
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
    comment TEXT,
    journal_ref TEXT,
    doi VARCHAR(100),
    pdf_downloaded BOOLEAN DEFAULT FALSE,
    pdf_processed BOOLEAN DEFAULT FALSE,
    embedding_generated BOOLEAN DEFAULT FALSE,
    processing_error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_papers_published_date ON papers (published_date DESC);
CREATE INDEX IF NOT EXISTS idx_papers_categories ON papers USING GIN (categories);
CREATE INDEX IF NOT EXISTS idx_papers_authors ON papers USING GIN (authors);

-- Chunks table with embeddings
CREATE TABLE IF NOT EXISTS paper_chunks (
    id SERIAL PRIMARY KEY,
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_tokens INTEGER,
    embedding vector(384),
    section_name VARCHAR(255),
    page_number INTEGER,
    char_start INTEGER,
    char_end INTEGER,
    has_math BOOLEAN DEFAULT FALSE,
    has_code BOOLEAN DEFAULT FALSE,
    has_references BOOLEAN DEFAULT FALSE,
    language VARCHAR(10) DEFAULT 'en',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (paper_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON paper_chunks
USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_chunks_paper_id ON paper_chunks (paper_id);
CREATE INDEX IF NOT EXISTS idx_chunks_text_trgm ON paper_chunks USING GIN (chunk_text gin_trgm_ops);

-- Authors table
CREATE TABLE IF NOT EXISTS authors (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    normalized_name TEXT,
    affiliation TEXT,
    orcid VARCHAR(50),
    email VARCHAR(255),
    UNIQUE (normalized_name)
);

CREATE INDEX IF NOT EXISTS idx_authors_name_trgm ON authors USING GIN (name gin_trgm_ops);

-- Paper-Authors junction
CREATE TABLE IF NOT EXISTS paper_authors (
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    author_id INTEGER REFERENCES authors(id) ON DELETE CASCADE,
    author_position INTEGER,
    is_corresponding BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (paper_id, author_id)
);

CREATE INDEX IF NOT EXISTS idx_paper_authors_author_id ON paper_authors (author_id);

-- Categories
CREATE TABLE IF NOT EXISTS categories (
    code VARCHAR(20) PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    parent_category VARCHAR(20)
);

-- Search history
CREATE TABLE IF NOT EXISTS search_history (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_embedding vector(384),
    result_count INTEGER,
    execution_time_ms INTEGER,
    filters JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_search_history_created_at ON search_history (created_at DESC);

-- Processing queue
CREATE TABLE IF NOT EXISTS processing_queue (
    id SERIAL PRIMARY KEY,
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    operation VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    priority INTEGER DEFAULT 0,
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_queue_status_priority ON processing_queue (status, priority DESC);

-- Auto-update trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_papers_updated_at ON papers;
CREATE TRIGGER update_papers_updated_at
    BEFORE UPDATE ON papers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
"""


def setup_database():
    """Create database schema."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute(SCHEMA_SQL)
    cursor.close()
    conn.close()
    print("Database schema created.")


# =============================================================================
# Section 5 — ArXiv Client (STUBS)
# =============================================================================

@dataclass
class ArxivPaper:
    """Structured representation of an ArXiv paper."""
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    primary_category: str
    published_date: datetime
    updated_date: datetime
    pdf_url: str
    comment: Optional[str] = None
    journal_ref: Optional[str] = None
    doi: Optional[str] = None


class ArxivClient:
    """Robust ArXiv API client with rate limiting and error handling."""

    def __init__(self, rate_limit_seconds: float = 3.0,
                 max_results_per_query: int = 100):
        self.rate_limit_seconds = rate_limit_seconds
        self.max_results_per_query = max_results_per_query
        self.last_request_time = 0.0

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - elapsed)
        self.last_request_time = time.time()

    def search_papers(self, query: str, max_results: int = 100,
                      sort_by=arxiv.SortCriterion.SubmittedDate,
                      sort_order=arxiv.SortOrder.Descending
                      ) -> Generator[ArxivPaper, None, None]:
        """Search ArXiv. STUB — implement or pull from companion repo."""
        pass

    def fetch_by_ids(self, arxiv_ids: List[str]) -> List[ArxivPaper]:
        """Fetch specific papers by ArXiv ID. STUB."""
        pass

    def search_recent_papers(self, categories: List[str],
                             days_back: int = 7) -> Generator[ArxivPaper, None, None]:
        """Fetch recent papers by category. STUB."""
        pass


# =============================================================================
# Section 5 — PDF Downloader (STUBS)
# =============================================================================

class PDFDownloader:
    """Manages PDF downloads with retry logic and organization."""

    def __init__(self, storage_path: str = "./data/pdfs",
                 max_retries: int = 3, timeout: int = 30):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.timeout = timeout

    def _get_pdf_path(self, arxiv_id: str, published_date: datetime) -> Path:
        """Generate organized storage path. STUB."""
        pass

    def download_pdf(self, pdf_url: str, arxiv_id: str,
                     published_date: datetime,
                     force: bool = False) -> Tuple[bool, Optional[Path], Optional[str]]:
        """Download PDF with retry logic. STUB."""
        pass

    def _validate_pdf(self, pdf_path: Path) -> bool:
        """Check if file is a valid PDF. STUB."""
        pass


# =============================================================================
# Section 5 — Paper Processor with _upsert_paper (PARTIALLY IMPLEMENTED)
# =============================================================================

class PaperProcessor:
    """Orchestrates the complete paper processing pipeline."""

    def __init__(self, db_config: dict, arxiv_client: ArxivClient,
                 pdf_downloader: PDFDownloader, max_workers: int = 4):
        self.db_config = db_config
        self.arxiv_client = arxiv_client
        self.pdf_downloader = pdf_downloader
        self.max_workers = max_workers

    def process_papers_batch(self, query: str, max_papers: int = 100,
                             skip_existing: bool = True) -> dict:
        """Process a batch of papers from search query. STUB."""
        pass

    def _upsert_paper(self, cursor, paper: ArxivPaper) -> int:
        """Insert or update paper metadata, return paper ID. IMPLEMENTED."""
        cursor.execute("""
            INSERT INTO papers (arxiv_id, title, abstract, authors, categories,
                                primary_category, published_date, updated_date,
                                pdf_url, comment, journal_ref, doi)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (arxiv_id) DO UPDATE SET title = EXCLUDED.title
            RETURNING id
        """, (paper.arxiv_id, paper.title, paper.abstract,
              paper.authors, paper.categories, paper.primary_category,
              paper.published_date, paper.updated_date,
              paper.pdf_url, paper.comment, paper.journal_ref, paper.doi))

        paper_db_id = cursor.fetchone()['id']

        # Sync relational author tables
        for author_name in paper.authors:
            cursor.execute("""
                INSERT INTO authors (name, normalized_name)
                VALUES (%s, %s)
                ON CONFLICT (normalized_name) DO NOTHING
                RETURNING id
            """, (author_name, author_name))

            res = cursor.fetchone()
            if res:
                author_id = res['id']
            else:
                cursor.execute("SELECT id FROM authors WHERE normalized_name = %s",
                               (author_name,))
                author_id = cursor.fetchone()['id']

            cursor.execute("""
                INSERT INTO paper_authors (paper_id, author_id)
                VALUES (%s, %s) ON CONFLICT DO NOTHING
            """, (paper_db_id, author_id))

        return paper_db_id

    def _process_queue(self, conn, stats: dict):
        """Process items from the queue. STUB."""
        pass


# =============================================================================
# Section 6 — PDF Extraction (STUBS)
# =============================================================================

@dataclass
class ExtractedPage:
    page_num: int
    text: str
    blocks: List[Dict]
    has_columns: bool
    has_math: bool
    has_tables: bool
    confidence: float


class PDFExtractor:
    """Advanced PDF text extraction for academic papers."""

    def __init__(self):
        pass

    def extract_paper_text(self, pdf_path: str) -> Dict[str, any]:
        """Extract complete text from PDF. STUB."""
        pass

    def _extract_page(self, page, page_num: int) -> ExtractedPage:
        pass

    def _detect_columns(self, blocks: Dict) -> bool:
        pass

    def _clean_extracted_text(self, text: str) -> str:
        pass


class TextChunker:
    """Intelligent text chunking for academic papers."""

    def __init__(self, target_chunk_size: int = 768, min_chunk_size: int = 256,
                 max_chunk_size: int = 1024, overlap_size: int = 128):
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size

    def chunk_paper(self, text: str, sections: List[Dict],
                    preserve_sections: bool = True) -> List[Dict]:
        """Chunk paper text intelligently. STUB."""
        pass

    def _chunk_text(self, text: str,
                    section_name: Optional[str] = None) -> List[Dict]:
        pass


# =============================================================================
# Section 7 — Embedding Generator (IMPLEMENTED: Singleton)
# =============================================================================

class EmbeddingGenerator:
    """Efficient embedding generation with singleton pattern."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmbeddingGenerator, cls).__new__(cls)
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

    def generate_embeddings(self, texts: List[str],
                            show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts. STUB."""
        pass

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding. STUB."""
        pass

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a search query. STUB."""
        pass


class EmbeddingPipeline:
    """Complete pipeline for generating and storing embeddings."""

    def __init__(self, db_config: dict, embedding_generator: EmbeddingGenerator,
                 batch_size: int = 100):
        self.db_config = db_config
        self.embedding_generator = embedding_generator
        self.batch_size = batch_size

    def process_paper(self, paper_id: int, chunks: List[Dict]) -> Dict[str, any]:
        """Process all chunks for a single paper. STUB."""
        pass

    def _store_chunks_with_embeddings(self, paper_id: int, chunks: List[Dict],
                                      embeddings: np.ndarray):
        """Store chunks and embeddings in database. STUB."""
        pass

    def process_pending_papers(self, limit: int = 10) -> Dict[str, any]:
        """Process papers needing embedding generation. STUB."""
        pass


# =============================================================================
# Section 8 — Search Engine (STUBS)
# =============================================================================

class SearchMode(Enum):
    VECTOR = "vector"
    HYBRID = "hybrid"
    KEYWORD = "keyword"


@dataclass
class SearchResult:
    paper_id: int
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    score: float
    matched_chunks: List[Dict]
    published_date: str
    categories: List[str]


class PaperSearchEngine:
    """Advanced search engine for academic papers."""

    def __init__(self, db_config: dict, embedding_generator: EmbeddingGenerator):
        self.db_config = db_config
        self.embedding_generator = embedding_generator

    def search(self, query: str, mode: SearchMode = SearchMode.HYBRID,
               limit: int = 10, filters: Optional[Dict] = None) -> List[SearchResult]:
        """Main search interface. STUB."""
        pass

    def _vector_search(self, query: str, limit: int,
                       filters: Optional[Dict]) -> List[SearchResult]:
        pass

    def _hybrid_search(self, query: str, limit: int,
                       filters: Optional[Dict]) -> List[SearchResult]:
        pass

    def _keyword_search(self, query: str, limit: int,
                        filters: Optional[Dict]) -> List[SearchResult]:
        pass

    def find_similar_papers(self, paper_id: int,
                            limit: int = 10) -> List[SearchResult]:
        pass


# =============================================================================
# Section 9 — CLI (STUBS)
# =============================================================================

def cli_main():
    """Entry point for the CLI. STUB — see companion repo for click-based CLI."""
    print("ArXiv Paper Search System")
    print("Run setup_database() to initialize schema.")
    print("See companion GitHub repo for full CLI implementation.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_database()
    else:
        cli_main()
