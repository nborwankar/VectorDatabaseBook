# Chapter 8: Complete Conversation Search and RAG System

Production-oriented RAG with conversation history, FastAPI server,
HTMX frontend, and Ollama for local LLM inference.

## Prerequisites

1. **PostgreSQL 15+** with **pgvector** extension
2. **Ollama** — https://ollama.ai
3. Create database: `createdb conversation_rag`

## Setup

```bash
python -m venv ch8_env
source ch8_env/bin/activate
pip install -r requirements.txt

# Initialize schema
python app.py setup

# Load sample documents
python app.py load-samples

# Start Ollama
ollama serve
ollama pull llama3.1:8b
```

## Run

### Web server (FastAPI + HTMX)
```bash
python app.py serve        # default port 8000
python app.py serve 9000   # custom port
```

Then visit http://localhost:8000

### CLI mode
```bash
python app.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | /api/health | Health check (Ollama status) |
| POST   | /api/sessions | Create conversation session |
| GET    | /api/sessions | List sessions |
| GET    | /api/sessions/{id}/messages | Get conversation history |
| POST   | /api/ask?session_id=X&question=Y | Ask a question (RAG) |
| POST   | /api/documents | Ingest a document |
| GET    | /api/search?q=X | Search document chunks |
| GET    | /api/search/conversations?q=X | Search past conversations |

## Architecture

1. **Documents** → chunk → embed → pgvector (HNSW index)
2. **Search**: Hybrid (semantic + keyword via ts_rank)
3. **Conversations**: Session-based, all messages embedded for cross-session search
4. **Generation**: Ollama with conversation history + retrieved chunks
5. **Frontend**: HTMX-powered, server-rendered HTML fragments
6. **Connection pool**: ThreadedConnectionPool for concurrent requests
