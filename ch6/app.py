"""
Chapter 6: Building a RAG System with SQLite VSS and Ollama
============================================================
A local, private RAG system combining SQLite-VSS for vector search
and Ollama for local LLM inference.

Dependencies: pip install sentence-transformers requests
Also requires: sqlite-vss binaries (vector0, vss0), Ollama running locally
"""

import sqlite3
import requests
import json
import time
import hashlib
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional


# =============================================================================
# 6.1 - Database Foundation
# =============================================================================

def setup_database(db_path='reddit_rag.db'):
    """Setup SQLite with VSS extension and create tables."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    conn.enable_load_extension(True)
    try:
        conn.load_extension("./vss0")
        conn.load_extension("./vector0")
        print("VSS extension loaded successfully")
    except Exception as e:
        print(f"Error loading VSS: {e}")
        print("Download from: https://github.com/asg017/sqlite-vss/releases")
        raise

    cursor = conn.cursor()

    # Posts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            post_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            title TEXT,
            content TEXT,
            subreddit TEXT,
            author TEXT,
            score INTEGER
        )
    """)

    # Chunks table with vector support
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS content_chunks (
            chunk_id INTEGER PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            post_id TEXT,
            chunk_index INTEGER,
            content TEXT,
            chunk_vector BLOB,
            FOREIGN KEY (post_id) REFERENCES posts(post_id)
        )
    """)

    # VSS virtual table for vector search
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vss USING vss0(
            chunk_vector(384),
            chunk_id INTEGER
        )
    """)

    # FTS5 for keyword search
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            content,
            content='content_chunks',
            content_rowid='chunk_id'
        )
    """)

    conn.commit()
    return conn


# =============================================================================
# 6.2 - Text Processing and Embedding Generation
# =============================================================================

embedding_model = None


def get_embedding_model():
    """Get or initialize the embedding model."""
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"Loaded embedding model (dimension: "
              f"{embedding_model.get_sentence_embedding_dimension()})")
    return embedding_model


def chunk_text(text, chunk_size=200, overlap=50):
    """Simple text chunking by word count."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def store_post_with_chunks(conn, post_data):
    """Store a post and its chunks with embeddings."""
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO posts (post_id, title, content, subreddit, author, score)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        post_data['post_id'], post_data['title'], post_data['content'],
        post_data['subreddit'], post_data['author'], post_data.get('score', 0)
    ))

    full_text = f"{post_data['title']}\n\n{post_data['content']}"
    chunks = chunk_text(full_text)
    model = get_embedding_model()

    for idx, chunk_content in enumerate(chunks):
        embedding = model.encode(chunk_content)

        cursor.execute("""
            INSERT INTO content_chunks (post_id, chunk_index, content, chunk_vector)
            VALUES (?, ?, ?, ?)
        """, (post_data['post_id'], idx, chunk_content, embedding.tobytes()))

        chunk_id = cursor.lastrowid

        # Add to VSS index
        vector_str = f"vss_create_vector({','.join(map(str, embedding))})"
        cursor.execute(f"""
            INSERT INTO chunk_vss (chunk_id, chunk_vector)
            VALUES (?, {vector_str})
        """, (chunk_id,))

        # Add to FTS index
        cursor.execute("""
            INSERT INTO chunks_fts (rowid, content) VALUES (?, ?)
        """, (chunk_id, chunk_content))

    conn.commit()
    print(f"Stored post {post_data['post_id']} with {len(chunks)} chunks")


# =============================================================================
# 6.3 - Hybrid Search
# =============================================================================

def hybrid_search(conn, query_text, limit=5, semantic_weight=0.7):
    """Perform hybrid search combining vector and keyword search."""
    cursor = conn.cursor()
    model = get_embedding_model()

    query_embedding = model.encode(query_text)
    vector_str = f"vss_create_vector({','.join(map(str, query_embedding))})"

    # --- Semantic search ---
    try:
        cursor.execute(f"""
            SELECT rowid, vss_cosine_distance(chunk_vector, {vector_str}) as distance
            FROM chunk_vss
            ORDER BY distance ASC
            LIMIT ?
        """, (limit * 2,))
        semantic_rows = cursor.fetchall()
    except Exception as e:
        print(f"Semantic search error: {e}")
        semantic_rows = []

    semantic_results = {}
    for row in semantic_rows:
        chunk_id = row[0]
        similarity = 1 - row[1]  # Convert distance to similarity
        semantic_results[chunk_id] = similarity

    # --- Keyword search ---
    keyword_results = {}
    try:
        cursor.execute("""
            SELECT rowid, bm25(chunks_fts) as score
            FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?
        """, (query_text, limit * 2))
        keyword_rows = cursor.fetchall()
        # BM25 scores are negative in SQLite; invert
        keyword_results = {row[0]: -row[1] for row in keyword_rows}
    except sqlite3.OperationalError as e:
        print(f"Warning: Keyword search failed: {e}")

    # --- Normalize keyword scores ---
    max_key = max(keyword_results.values()) if keyword_results else 1.0
    if max_key > 0:
        keyword_results = {k: v / max_key for k, v in keyword_results.items()}

    # --- Merge: semantic * 0.7 + keyword * 0.3 ---
    merged = {}
    keyword_weight = 1.0 - semantic_weight
    for cid, score in semantic_results.items():
        merged[cid] = score * semantic_weight
    for cid, score in keyword_results.items():
        merged[cid] = merged.get(cid, 0) + (score * keyword_weight)

    sorted_ids = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:limit]

    # --- Fetch full chunk data ---
    results = []
    for chunk_id, score in sorted_ids:
        cursor.execute("""
            SELECT chunk_id, post_id, content FROM content_chunks WHERE chunk_id = ?
        """, (chunk_id,))
        row = cursor.fetchone()
        if row:
            results.append({
                'chunk_id': row['chunk_id'],
                'post_id': row['post_id'],
                'content': row['content'],
                'score': score
            })

    return results


# =============================================================================
# 6.4 - LLM Integration with Ollama
# =============================================================================

def call_ollama(prompt, model="llama3.1:8b", temperature=0.1):
    """Simple Ollama API call."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_ctx": 4096
        }
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()['response']
    except Exception as e:
        print(f"Ollama error: {e}")
        return f"Error calling Ollama: {str(e)}"


def test_ollama():
    """Test if Ollama is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        models = response.json()
        print("Available Ollama models:",
              [m['name'] for m in models.get('models', [])])
        return True
    except Exception:
        print("Ollama not running. Start with: ollama serve")
        return False


# =============================================================================
# 6.5 - RAG Pipeline
# =============================================================================

def format_context(chunks, conn):
    """Format retrieved chunks for the LLM."""
    cursor = conn.cursor()
    formatted_chunks = []

    for i, chunk in enumerate(chunks, 1):
        cursor.execute("""
            SELECT title, subreddit, author, score
            FROM posts WHERE post_id = ?
        """, (chunk['post_id'],))

        post = cursor.fetchone()
        if post:
            formatted_chunks.append(f"""
SOURCE {i}:
From r/{post['subreddit']} by u/{post['author']} (Score: {post['score']})
Title: {post['title']}

Content:
{chunk['content']}
""")

    return "\n---\n".join(formatted_chunks)


def answer_question(conn, question, num_chunks=5):
    """Complete RAG pipeline to answer a question."""
    print(f"\nQuestion: {question}")

    start_time = time.time()
    chunks = hybrid_search(conn, question, limit=num_chunks)
    retrieval_time = (time.time() - start_time) * 1000

    if not chunks:
        return "No relevant information found in the database."

    print(f"Retrieved {len(chunks)} chunks in {retrieval_time:.1f}ms")

    context = format_context(chunks, conn)

    prompt = f"""You are a helpful assistant. Answer the user's question using \
ONLY the retrieved information provided below.

Instructions:
1. If the information is not in the context, state that you do not know.
2. Do not use outside knowledge.
3. For every claim you make, cite the source number (e.g., [Source 1]).

RETRIEVED INFORMATION:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    start_time = time.time()
    answer = call_ollama(prompt)
    generation_time = (time.time() - start_time) * 1000

    print(f"Generated answer in {generation_time:.1f}ms")
    print(f"Total time: {retrieval_time + generation_time:.1f}ms")

    return answer


# =============================================================================
# 6.6 - Sample Data and Demo
# =============================================================================

def load_sample_data(conn):
    """Load sample Reddit data for testing."""
    sample_posts = [
        {
            'post_id': 'post001',
            'title': 'ELI5: What is machine learning?',
            'content': """Machine learning is like teaching a computer to recognize \
patterns by showing it many examples. Instead of programming exact rules, you feed \
it data and it learns the patterns itself. For example, to teach it to recognize \
cats, you show it thousands of cat pictures until it learns what makes a cat a cat. \
It's used in spam filters, recommendation systems, and voice assistants.""",
            'subreddit': 'explainlikeimfive',
            'author': 'curious_user',
            'score': 245
        },
        {
            'post_id': 'post002',
            'title': 'Best Python libraries for beginners?',
            'content': """I recommend starting with: NumPy for numerical computing, \
Pandas for data manipulation, Matplotlib for basic plotting, Requests for HTTP \
requests, and Flask for simple web apps. These libraries cover most beginner needs \
and have excellent documentation. Don't try to learn them all at once - start with \
one based on your project needs.""",
            'subreddit': 'learnpython',
            'author': 'python_mentor',
            'score': 189
        },
        {
            'post_id': 'post003',
            'title': 'How do vector databases work?',
            'content': """Vector databases store data as high-dimensional vectors \
(lists of numbers) that represent the semantic meaning of the data. When you search, \
your query is converted to a vector and the database finds the most similar vectors \
using distance metrics like cosine similarity. This enables semantic search where you \
find content by meaning rather than exact keyword matches. Popular vector databases \
include Pinecone, Weaviate, and pgvector for PostgreSQL.""",
            'subreddit': 'programming',
            'author': 'db_expert',
            'score': 156
        }
    ]

    print("Loading sample data...")
    for post in sample_posts:
        store_post_with_chunks(conn, post)
    print(f"Loaded {len(sample_posts)} sample posts")


def main():
    """Main demonstration of the RAG system."""
    print("=== Minimal RAG System with SQLite VSS and Ollama ===\n")

    print("1. Setting up database...")
    conn = setup_database()

    print("\n2. Checking Ollama...")
    if not test_ollama():
        print("Please start Ollama first: ollama serve")
        print("Then pull a model: ollama pull llama3.1:8b")
        return

    print("\n3. Loading sample data...")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM posts")
    if cursor.fetchone()[0] == 0:
        load_sample_data(conn)
    else:
        print("Data already loaded")

    print("\n4. Demonstrating RAG system...")
    demo_questions = [
        "What is machine learning and how does it work?",
        "What Python libraries should a beginner learn?",
        "How do vector databases enable semantic search?"
    ]

    for question in demo_questions[:1]:
        answer = answer_question(conn, question)
        print(f"\nAnswer: {answer}\n")
        print("=" * 50)

    # Interactive mode
    print("\n5. Interactive Q&A (type 'quit' to exit)")
    print("-" * 50)

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if not question:
            continue
        answer = answer_question(conn, question)
        print(f"\nAnswer: {answer}")

    conn.close()
    print("\nGoodbye!")


def quick_start():
    """Quick start for testing individual components."""
    conn = setup_database()

    print("Testing hybrid search for 'machine learning'...")
    results = hybrid_search(conn, "machine learning", limit=3)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   Content: {result['content'][:150]}...")

    conn.close()


if __name__ == "__main__":
    main()
    # quick_start()
