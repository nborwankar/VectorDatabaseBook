# Chapter 6: RAG System with SQLite-VSS and Ollama

Local, private RAG combining SQLite-VSS vector search with Ollama LLM inference.

## Prerequisites

1. **sqlite-vss binaries** — https://github.com/asg017/sqlite-vss/releases
   - Place `vector0.so` and `vss0.so` in the same directory as `app.py`
   - Not officially supported on Windows — use WSL2

2. **Ollama** — https://ollama.ai
   ```bash
   ollama serve          # Start the server
   ollama pull llama3.1:8b  # Pull a model
   ```

## Setup

```bash
python -m venv ch6_env
source ch6_env/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Loads sample data, runs a demo question through the RAG pipeline, then enters interactive Q&A mode.

## Architecture

1. SQLite + VSS extension for vector storage (384-dim, all-MiniLM-L6-v2)
2. FTS5 for keyword search (BM25 scoring)
3. Hybrid search: 70% semantic + 30% keyword (normalized + merged)
4. Ollama (llama3.1:8b) for answer generation with low temperature (0.1)
