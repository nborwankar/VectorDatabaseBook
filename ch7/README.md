# Chapter 7: Scientific RAG with PostgreSQL pgvector + Ollama

RAG system for scientific paper Q&A. Hybrid search (semantic + keyword) over
chunked papers with Ollama for local generation.

## Prerequisites

1. **PostgreSQL 15+** with **pgvector** extension
2. **Ollama** — https://ollama.ai
3. Database from Chapter 5 (or run setup to create fresh)

## Setup

```bash
python -m venv ch7_env
source ch7_env/bin/activate
pip install -r requirements.txt

# Initialize database schema
python app.py setup

# Load sample papers for testing
python app.py load-samples

# Start Ollama
ollama serve
ollama pull llama3.1:8b
```

## Configuration

Environment variables (or edit DB_CONFIG / OLLAMA_* in app.py):
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- `OLLAMA_URL` (default: http://localhost:11434)
- `OLLAMA_MODEL` (default: llama3.1:8b)

## Run

```bash
python app.py
```

## Architecture

1. **Ingestion**: Paper text → section-aware chunking → embeddings → pgvector
2. **Search**: Hybrid (70% cosine similarity via HNSW + 30% keyword via ts_rank)
3. **Generation**: Ollama with grounded prompt + source citations
4. **Analytics**: Query logging with embeddings for search quality analysis
