# Chapter 5: ArXiv Paper Search with PostgreSQL pgvector

**NOTE**: This chapter is an architecture scaffold. Most methods are stubs (`pass`).
Full implementations are in the companion GitHub repo mentioned in the book preface.

The SQL schema and `_upsert_paper` method are fully implemented.

## Prerequisites

1. **PostgreSQL 15+** with **pgvector** extension
2. Create the database: `createdb arxiv_papers`

## Setup

```bash
python -m venv ch5_env
source ch5_env/bin/activate
pip install -r requirements.txt

# Initialize schema
python app.py setup
# OR
psql -d arxiv_papers -f schema.sql
```

## Configuration

Set environment variables or edit DB_CONFIG in app.py:
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`

## What's complete vs. stub

**Complete**: SQL schema, `_upsert_paper`, `EmbeddingGenerator` singleton, all dataclasses
**Stubs**: ArxivClient methods, PDFDownloader, PDFExtractor, TextChunker, search methods, CLI

## Docker

The chapter also includes Docker Compose config (see book text) using `pgvector/pgvector:pg15`.
