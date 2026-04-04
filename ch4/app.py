"""
Chapter 4: Semantic Search with SQLite3
========================================
Reddit personal knowledge base with SQLite-VSS vector search.

Dependencies: pip install praw sentence-transformers numpy
Also requires sqlite-vss binaries (vector0, vss0) - see README.md
"""

import sqlite3
import json
import re
import time
import numpy as np
from typing import List, Dict, Any, Optional, Generator, Union
from dataclasses import dataclass
from datetime import datetime

import praw
from prawcore.exceptions import PrawcoreException
from sentence_transformers import SentenceTransformer


# =============================================================================
# 4.2.2 - Verify VSS Installation
# =============================================================================

def verify_vss_installation(extension_path: str = ".") -> bool:
    """
    Verify sqlite-vss extension loads and works correctly.

    Args:
        extension_path: Directory containing vss0 and vector0 extensions

    Returns:
        True if verification succeeds, False otherwise
    """
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)

    try:
        conn.load_extension(f"{extension_path}/vector0")
        conn.load_extension(f"{extension_path}/vss0")
        conn.enable_load_extension(False)

        result = conn.execute("""
            SELECT vss_version(),
                   vss_distance_l2(
                       vector_from_json('[1.0, 0.0, 0.0]'),
                       vector_from_json('[0.0, 1.0, 0.0]')
                   )
        """).fetchone()

        version, distance = result
        expected_distance = 1.414  # sqrt(2) for orthogonal unit vectors

        print(f"sqlite-vss version: {version}")
        print(f"L2 distance test: {distance:.3f} (expected ~{expected_distance:.3f})")

        conn.close()
        return abs(distance - expected_distance) < 0.01

    except Exception as e:
        print(f"Verification failed: {e}")
        conn.close()
        return False


# =============================================================================
# 4.2.3 - Operational Pragmas
# =============================================================================

def configure_connection(conn: sqlite3.Connection) -> None:
    """Apply recommended pragmas for performance and correctness."""
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=ON;")


# =============================================================================
# 4.3.2 - Database Creation and Schema
# =============================================================================

def create_database(db_path: str, extension_path: str = ".") -> sqlite3.Connection:
    """
    Create the Reddit knowledge base database with proper schema.

    Args:
        db_path: Path to the SQLite database file
        extension_path: Directory containing vss0 and vector0 extensions

    Returns:
        Database connection with extensions loaded
    """
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(f"{extension_path}/vector0")
    conn.load_extension(f"{extension_path}/vss0")
    conn.enable_load_extension(False)

    configure_connection(conn)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY,
            post_id TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            selftext TEXT,
            url TEXT,
            subreddit TEXT NOT NULL,
            author TEXT,
            score INTEGER DEFAULT 0,
            num_comments INTEGER DEFAULT 0,
            created_utc INTEGER NOT NULL,
            embedding BLOB,
            embedding_model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_subreddit ON posts(subreddit)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(created_utc)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_score ON posts(score)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_post_id ON posts(post_id)")

    conn.commit()
    return conn


def create_vector_index(conn: sqlite3.Connection, dimension: int) -> None:
    """
    Create the VSS virtual table for vector similarity search.

    Args:
        conn: Database connection with vss extension loaded
        dimension: Embedding vector dimension (e.g., 384 for all-MiniLM-L6-v2)
    """
    conn.execute("DROP TABLE IF EXISTS posts_vss")
    conn.execute(f"""
        CREATE VIRTUAL TABLE posts_vss USING vss0(
            embedding({dimension})
        )
    """)
    conn.commit()
    print(f"Created vector index with dimension {dimension}")


# =============================================================================
# 4.4.2 - Reddit Client (PRAW)
# =============================================================================

class RedditClient:
    """Client for fetching Reddit content with rate limiting and error handling."""

    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        assert self.reddit.read_only, "Client should be read-only"
        self.request_delay = 1.0
        self.last_request_time = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request_time = time.time()

    def _retry_with_backoff(self, func, max_retries: int = 3) -> Any:
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                return func()
            except PrawcoreException as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    wait_time = min(2 ** attempt * 10, 300)
                    print(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1})")
                    time.sleep(wait_time)
                    self.request_delay *= 1.5
                else:
                    raise
        raise PrawcoreException("Max retries exceeded")

    def get_subreddit_posts(
        self,
        subreddit_name: str,
        sort: str = "hot",
        limit: int = 100,
        time_filter: str = "all"
    ) -> Generator[Dict[str, Any], None, None]:
        subreddit = self.reddit.subreddit(subreddit_name)

        if sort == "hot":
            posts = subreddit.hot(limit=limit)
        elif sort == "new":
            posts = subreddit.new(limit=limit)
        elif sort == "top":
            posts = subreddit.top(time_filter=time_filter, limit=limit)
        elif sort == "rising":
            posts = subreddit.rising(limit=limit)
        else:
            raise ValueError(f"Unknown sort method: {sort}")

        def fetch_batch():
            return list(posts)

        fetched_posts = self._retry_with_backoff(fetch_batch)

        for submission in fetched_posts:
            yield {
                "post_id": submission.id,
                "title": submission.title,
                "selftext": submission.selftext,
                "url": submission.url,
                "subreddit": submission.subreddit.display_name,
                "author": str(submission.author) if submission.author else "[deleted]",
                "score": submission.score,
                "num_comments": submission.num_comments,
                "created_utc": int(submission.created_utc),
                "is_self": submission.is_self
            }


# =============================================================================
# 4.5.1 - Text Preprocessing
# =============================================================================

class TextPreprocessor:
    """Clean and normalize Reddit text content for embedding."""

    def __init__(self):
        self.url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
        self.reddit_link_pattern = re.compile(r'(?<!\w)/r/\w+|(?<!\w)/u/\w+')
        self.markdown_link_pattern = re.compile(r'\[([^\]]+)\]\([^)]+\)')
        self.whitespace_pattern = re.compile(r'\s+')

    def clean_markdown(self, text: str) -> str:
        text = self.markdown_link_pattern.sub(r'\1', text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)       # Italic
        text = re.sub(r'~~([^~]+)~~', r'\1', text)       # Strikethrough
        text = re.sub(r'`([^`]+)`', r'\1', text)         # Inline code
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # Headers
        text = re.sub(r'^[>\-*]\s*', '', text, flags=re.MULTILINE)  # Quotes/lists
        return text

    def handle_urls(self, text: str) -> str:
        return self.url_pattern.sub('[URL]', text)

    def clean_reddit_artifacts(self, text: str) -> str:
        text = re.sub(r'\[removed\]|\[deleted\]', '', text)
        text = self.reddit_link_pattern.sub('', text)
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&nbsp;', ' ')
        return text

    def normalize_whitespace(self, text: str) -> str:
        text = self.whitespace_pattern.sub(' ', text)
        return text.strip()

    def process(self, text: Optional[str]) -> str:
        if not text:
            return ""
        text = self.clean_markdown(text)
        text = self.handle_urls(text)
        text = self.clean_reddit_artifacts(text)
        text = self.normalize_whitespace(text)
        return text


class ContentPreparer:
    """Prepare Reddit content for embedding generation."""

    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def prepare_post(self, post: dict) -> str:
        title = self.preprocessor.process(post.get("title", ""))
        selftext = self.preprocessor.process(post.get("selftext", ""))
        if selftext:
            return f"{title}. {selftext}"
        return title

    def is_quality_content(self, text: str, min_length: int = 20, min_words: int = 3) -> bool:
        if len(text) < min_length:
            return False
        if len(text.split()) < min_words:
            return False
        if text in ["[URL]", "", " "]:
            return False
        return True


# =============================================================================
# 4.6.1 - Embedding Generator
# =============================================================================

class EmbeddingGenerator:
    """Generate embeddings using SentenceTransformers models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.device = self._select_device()
        self.model.to(self.device)
        print(f"Loaded {model_name} (dim={self.dimension}) on {self.device}")

    def _select_device(self) -> str:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32,
               show_progress: bool = False) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(
            texts, batch_size=batch_size,
            show_progress_bar=show_progress, convert_to_numpy=True
        )

    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> bytes:
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def deserialize_embedding(blob: bytes, dimension: int) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32).reshape(dimension)


# =============================================================================
# 4.6.2 - Knowledge Base Storage
# =============================================================================

class KnowledgeBaseStorage:
    """Handle storage and retrieval of Reddit content with embeddings."""

    def __init__(self, db_path: str, extension_path: str = ".",
                 embedding_dim: int = 384):
        self.db_path = db_path
        self.extension_path = extension_path
        self.embedding_dim = embedding_dim
        self.conn = self._connect()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        conn.load_extension(f"{self.extension_path}/vector0")
        conn.load_extension(f"{self.extension_path}/vss0")
        conn.enable_load_extension(False)
        configure_connection(conn)
        conn.row_factory = sqlite3.Row
        return conn

    def insert_post(self, post: Dict[str, Any], embedding: Optional[np.ndarray],
                    model_name: str) -> int:
        embedding_blob = None
        if embedding is not None:
            embedding_blob = EmbeddingGenerator.serialize_embedding(embedding)

        try:
            cursor = self.conn.execute("""
                INSERT INTO posts
                (post_id, title, selftext, url, subreddit, author,
                 score, num_comments, created_utc, embedding, embedding_model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(post_id) DO UPDATE SET
                    title = excluded.title, selftext = excluded.selftext,
                    url = excluded.url, subreddit = excluded.subreddit,
                    author = excluded.author, score = excluded.score,
                    num_comments = excluded.num_comments,
                    created_utc = excluded.created_utc,
                    embedding = excluded.embedding,
                    embedding_model = excluded.embedding_model
                RETURNING id
            """, (
                post["post_id"], post["title"], post.get("selftext", ""),
                post.get("url", ""), post["subreddit"],
                post.get("author", "[unknown]"), post.get("score", 0),
                post.get("num_comments", 0), post["created_utc"],
                embedding_blob, model_name
            ))
            row_id = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            # Fallback for older SQLite without RETURNING
            self.conn.execute("""
                INSERT INTO posts
                (post_id, title, selftext, url, subreddit, author,
                 score, num_comments, created_utc, embedding, embedding_model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(post_id) DO UPDATE SET
                    title = excluded.title, selftext = excluded.selftext,
                    url = excluded.url, subreddit = excluded.subreddit,
                    author = excluded.author, score = excluded.score,
                    num_comments = excluded.num_comments,
                    created_utc = excluded.created_utc,
                    embedding = excluded.embedding,
                    embedding_model = excluded.embedding_model
            """, (
                post["post_id"], post["title"], post.get("selftext", ""),
                post.get("url", ""), post["subreddit"],
                post.get("author", "[unknown]"), post.get("score", 0),
                post.get("num_comments", 0), post["created_utc"],
                embedding_blob, model_name
            ))
            row_id = self.conn.execute(
                "SELECT id FROM posts WHERE post_id = ?", (post["post_id"],)
            ).fetchone()[0]

        self.conn.commit()
        return row_id

    def insert_posts_batch(self, posts: List[Dict[str, Any]],
                           embeddings: np.ndarray, model_name: str) -> int:
        data = []
        for post, embedding in zip(posts, embeddings):
            embedding_blob = EmbeddingGenerator.serialize_embedding(embedding)
            data.append((
                post["post_id"], post["title"], post.get("selftext", ""),
                post.get("url", ""), post["subreddit"],
                post.get("author", "[unknown]"), post.get("score", 0),
                post.get("num_comments", 0), post["created_utc"],
                embedding_blob, model_name
            ))

        self.conn.executemany("""
            INSERT INTO posts
            (post_id, title, selftext, url, subreddit, author,
             score, num_comments, created_utc, embedding, embedding_model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(post_id) DO UPDATE SET
                title = excluded.title, selftext = excluded.selftext,
                url = excluded.url, subreddit = excluded.subreddit,
                author = excluded.author, score = excluded.score,
                num_comments = excluded.num_comments,
                created_utc = excluded.created_utc,
                embedding = excluded.embedding,
                embedding_model = excluded.embedding_model
        """, data)

        self.conn.commit()
        return len(data)

    def get_post_count(self) -> int:
        result = self.conn.execute("SELECT COUNT(*) FROM posts").fetchone()
        return result[0]

    def close(self) -> None:
        self.conn.close()


# =============================================================================
# 4.6.3 - Ingestion Pipeline
# =============================================================================

class RedditIngestionPipeline:
    """End-to-end pipeline for ingesting Reddit content."""

    def __init__(self, reddit_client: RedditClient,
                 storage: KnowledgeBaseStorage,
                 embedding_generator: EmbeddingGenerator):
        self.reddit = reddit_client
        self.storage = storage
        self.embedder = embedding_generator
        self.preparer = ContentPreparer()

    def ingest_subreddit(self, subreddit_name: str, sort: str = "top",
                         limit: int = 100, time_filter: str = "month",
                         batch_size: int = 32) -> dict:
        stats = {"fetched": 0, "processed": 0, "skipped": 0}
        posts_batch = []
        texts_batch = []

        for post in self.reddit.get_subreddit_posts(
            subreddit_name, sort=sort, limit=limit, time_filter=time_filter
        ):
            stats["fetched"] += 1
            text = self.preparer.prepare_post(post)

            if not self.preparer.is_quality_content(text):
                stats["skipped"] += 1
                continue

            posts_batch.append(post)
            texts_batch.append(text)

            if len(posts_batch) >= batch_size:
                self._process_batch(posts_batch, texts_batch)
                stats["processed"] += len(posts_batch)
                posts_batch = []
                texts_batch = []

        if posts_batch:
            self._process_batch(posts_batch, texts_batch)
            stats["processed"] += len(posts_batch)

        return stats

    def _process_batch(self, posts: List[Dict], texts: List[str]) -> None:
        embeddings = self.embedder.encode(texts)
        self.storage.insert_posts_batch(posts, embeddings, self.embedder.model_name)


# =============================================================================
# 4.7.2 - Vector Index Manager
# =============================================================================

class VectorIndexManager:
    """Manage the VSS vector index."""

    def __init__(self, storage: KnowledgeBaseStorage):
        self.storage = storage
        self.conn = storage.conn
        self.dimension = storage.embedding_dim

    def create_index(self, factory_string: str = "Flat") -> None:
        self.conn.execute("DROP TABLE IF EXISTS posts_vss")

        if factory_string == "Flat":
            self.conn.execute(f"""
                CREATE VIRTUAL TABLE posts_vss USING vss0(
                    embedding({self.dimension})
                )
            """)
        else:
            self.conn.execute(f"""
                CREATE VIRTUAL TABLE posts_vss USING vss0(
                    embedding({self.dimension}) factory="{factory_string}"
                )
            """)

        self.conn.commit()
        print(f"Created vector index: {factory_string}, dimension={self.dimension}")

    def populate_index(self, batch_size: int = 500) -> int:
        total = self.conn.execute(
            "SELECT COUNT(*) FROM posts WHERE embedding IS NOT NULL"
        ).fetchone()[0]

        if total == 0:
            print("No embeddings to index")
            return 0

        indexed = 0
        offset = 0

        while offset < total:
            rows = self.conn.execute("""
                SELECT id, embedding FROM posts
                WHERE embedding IS NOT NULL
                ORDER BY id LIMIT ? OFFSET ?
            """, (batch_size, offset)).fetchall()

            if not rows:
                break

            insert_data = []
            for row in rows:
                row_id = row["id"]
                embedding = EmbeddingGenerator.deserialize_embedding(
                    row["embedding"], self.dimension
                )
                vector_json = json.dumps(embedding.tolist())
                insert_data.append((row_id, vector_json))

            self.conn.executemany("""
                INSERT INTO posts_vss (rowid, embedding)
                VALUES (?, vector_from_json(?))
            """, insert_data)

            self.conn.commit()
            indexed += len(rows)
            offset += batch_size

            if indexed % 1000 == 0 or indexed == total:
                print(f"Indexed {indexed}/{total} vectors")

        return indexed

    def rebuild_index(self, factory_string: str = "Flat") -> int:
        self.create_index(factory_string)
        return self.populate_index()


# =============================================================================
# 4.8.1 - Search Result Container
# =============================================================================

@dataclass
class SearchResult:
    """Container for a single search result."""
    rowid: int
    post_id: str
    title: str
    selftext: str
    subreddit: str
    author: str
    score: int
    num_comments: int
    created_utc: int
    distance: float

    @property
    def similarity_score(self) -> float:
        return 1.0 / (1.0 + self.distance)

    @property
    def created_date(self) -> str:
        return datetime.utcfromtimestamp(self.created_utc).strftime("%Y-%m-%d")


# =============================================================================
# 4.8.2 - Semantic Search Engine
# =============================================================================

class SemanticSearchEngine:
    """Semantic search over Reddit knowledge base."""

    def __init__(self, storage: KnowledgeBaseStorage,
                 embedding_generator: EmbeddingGenerator,
                 candidate_multiplier: int = 10):
        self.storage = storage
        self.conn = storage.conn
        self.embedder = embedding_generator
        self.preparer = ContentPreparer()
        self.candidate_multiplier = candidate_multiplier

    def search(self, query: str, limit: int = 10,
               subreddits: Optional[List[str]] = None,
               min_score: Optional[int] = None,
               after_date: Optional[datetime] = None,
               before_date: Optional[datetime] = None) -> List[SearchResult]:

        clean_query = self.preparer.preprocessor.process(query)
        query_embedding = self.embedder.encode(clean_query)[0]
        query_vector_json = json.dumps(query_embedding.tolist())

        k = max(limit * self.candidate_multiplier, limit)

        sql = """
            WITH candidates AS (
                SELECT rowid, distance
                FROM posts_vss
                WHERE vss_search(embedding, vector_from_json(?))
                ORDER BY distance ASC
                LIMIT ?
            )
            SELECT
                p.id, p.post_id, p.title, p.selftext, p.subreddit,
                p.author, p.score, p.num_comments, p.created_utc,
                c.distance
            FROM candidates c
            INNER JOIN posts p ON p.id = c.rowid
            WHERE 1=1
        """
        params: List = [query_vector_json, k]

        if subreddits:
            placeholders = ",".join("?" * len(subreddits))
            sql += f" AND p.subreddit IN ({placeholders})"
            params.extend(subreddits)

        if min_score is not None:
            sql += " AND p.score >= ?"
            params.append(min_score)

        if after_date is not None:
            sql += " AND p.created_utc >= ?"
            params.append(int(after_date.timestamp()))

        if before_date is not None:
            sql += " AND p.created_utc <= ?"
            params.append(int(before_date.timestamp()))

        sql += " ORDER BY c.distance ASC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()

        return [SearchResult(
            rowid=row["id"], post_id=row["post_id"], title=row["title"],
            selftext=row["selftext"], subreddit=row["subreddit"],
            author=row["author"], score=row["score"],
            num_comments=row["num_comments"], created_utc=row["created_utc"],
            distance=row["distance"]
        ) for row in rows]

    def search_with_facets(self, query: str, limit: int = 50) -> dict:
        results = self.search(query, limit=limit)
        subreddit_counts = {}
        score_ranges = {"low": 0, "medium": 0, "high": 0}

        for r in results:
            subreddit_counts[r.subreddit] = subreddit_counts.get(r.subreddit, 0) + 1
            if r.score < 10:
                score_ranges["low"] += 1
            elif r.score < 100:
                score_ranges["medium"] += 1
            else:
                score_ranges["high"] += 1

        return {
            "results": results,
            "facets": {
                "subreddits": dict(sorted(subreddit_counts.items(),
                                          key=lambda x: x[1], reverse=True)),
                "score_ranges": score_ranges
            },
            "total": len(results)
        }

    def find_similar_posts(self, post_id: str, limit: int = 10,
                           exclude_same_subreddit: bool = False) -> List[SearchResult]:
        row = self.conn.execute(
            "SELECT id, embedding, subreddit FROM posts WHERE post_id = ?",
            (post_id,)
        ).fetchone()

        if not row or not row["embedding"]:
            return []

        embedding = EmbeddingGenerator.deserialize_embedding(
            row["embedding"], self.embedder.dimension
        )
        vector_json = json.dumps(embedding.tolist())
        source_subreddit = row["subreddit"]
        source_id = row["id"]

        sql = """
            SELECT
                p.id, p.post_id, p.title, p.selftext, p.subreddit,
                p.author, p.score, p.num_comments, p.created_utc,
                vss.distance
            FROM posts_vss vss
            INNER JOIN posts p ON p.id = vss.rowid
            WHERE vss_search(vss.embedding, vector_from_json(?))
            AND p.id != ?
        """
        params: List = [vector_json, source_id]

        if exclude_same_subreddit:
            sql += " AND p.subreddit != ?"
            params.append(source_subreddit)

        sql += " ORDER BY vss.distance ASC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()

        return [SearchResult(
            rowid=r["id"], post_id=r["post_id"], title=r["title"],
            selftext=r["selftext"], subreddit=r["subreddit"],
            author=r["author"], score=r["score"],
            num_comments=r["num_comments"], created_utc=r["created_utc"],
            distance=r["distance"]
        ) for r in rows]

    def cross_subreddit_analysis(self, query: str,
                                 limit_per_subreddit: int = 5) -> dict:
        results = self.search(query, limit=100)
        by_subreddit = {}
        for r in results:
            if r.subreddit not in by_subreddit:
                by_subreddit[r.subreddit] = []
            if len(by_subreddit[r.subreddit]) < limit_per_subreddit:
                by_subreddit[r.subreddit].append(r)

        subreddit_relevance = {
            sub: sum(r.similarity_score for r in posts)
            for sub, posts in by_subreddit.items()
        }
        sorted_subreddits = sorted(
            subreddit_relevance.keys(),
            key=lambda x: subreddit_relevance[x], reverse=True
        )

        return {
            "subreddits": sorted_subreddits,
            "posts_by_subreddit": {sub: by_subreddit[sub] for sub in sorted_subreddits},
            "relevance_scores": subreddit_relevance
        }


# =============================================================================
# 4.9 - Main: Putting It All Together
# =============================================================================

def main():
    """Complete example: ingest Reddit content and perform semantic search."""

    # Configuration
    DB_PATH = "reddit_knowledge.db"
    EXTENSION_PATH = "."  # Directory containing vss0 and vector0
    MODEL_NAME = "all-MiniLM-L6-v2"

    # Reddit credentials (use environment variables in production)
    REDDIT_CLIENT_ID = "your_client_id"
    REDDIT_CLIENT_SECRET = "your_client_secret"
    REDDIT_USER_AGENT = "python:reddit_kb:v1.0 (by /u/your_username)"

    print("Initializing components...")

    conn = create_database(DB_PATH, EXTENSION_PATH)
    conn.close()

    embedder = EmbeddingGenerator(MODEL_NAME)

    storage = KnowledgeBaseStorage(DB_PATH, EXTENSION_PATH,
                                   embedding_dim=embedder.dimension)

    reddit = RedditClient(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT)

    pipeline = RedditIngestionPipeline(reddit, storage, embedder)

    print("\nIngesting content...")
    subreddits = ["MachineLearning", "Python", "DataScience"]

    for subreddit in subreddits:
        print(f"\nFetching from r/{subreddit}...")
        stats = pipeline.ingest_subreddit(
            subreddit, sort="top", limit=100, time_filter="month"
        )
        print(f"  Fetched: {stats['fetched']}, "
              f"Processed: {stats['processed']}, "
              f"Skipped: {stats['skipped']}")

    print(f"\nTotal posts in database: {storage.get_post_count()}")

    print("\nBuilding vector index...")
    index_manager = VectorIndexManager(storage)
    index_manager.create_index()
    indexed = index_manager.populate_index()
    print(f"Indexed {indexed} vectors")

    search_engine = SemanticSearchEngine(storage, embedder)

    print("\n" + "=" * 60)
    print("SEMANTIC SEARCH EXAMPLES")
    print("=" * 60)

    query = "deep learning for edge devices"
    print(f"\nQuery: '{query}'")
    print("-" * 40)

    results = search_engine.search(query, limit=5)
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.subreddit}] {r.title[:60]}...")
        print(f"   Score: {r.score} | Similarity: {r.similarity_score:.3f} "
              f"| Distance: {r.distance:.3f}")

    query = "neural network optimization"
    print(f"\nQuery: '{query}' (filtered to r/MachineLearning, score >= 50)")
    print("-" * 40)

    results = search_engine.search(
        query, limit=5, subreddits=["MachineLearning"], min_score=50
    )
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.title[:60]}...")
        print(f"   Score: {r.score} | Date: {r.created_date}")

    query = "python best practices"
    print(f"\nCross-subreddit analysis: '{query}'")
    print("-" * 40)

    analysis = search_engine.cross_subreddit_analysis(query)
    for sub in analysis["subreddits"][:3]:
        relevance = analysis["relevance_scores"][sub]
        print(f"\nr/{sub} (relevance: {relevance:.2f}):")
        for r in analysis["posts_by_subreddit"][sub][:2]:
            print(f"  - {r.title[:50]}...")

    storage.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
