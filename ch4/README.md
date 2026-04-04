# Chapter 4: Semantic Search with SQLite3

Reddit personal knowledge base with SQLite-VSS vector search.

## Prerequisites

1. **sqlite-vss binaries** — Download from https://github.com/asg017/sqlite-vss/releases
   - Extract `vector0.so` and `vss0.so` (Linux), `.dylib` (macOS), or `.dll` (Windows)
   - Place them in the same directory as `app.py` (or set `EXTENSION_PATH`)
   - **Note**: sqlite-vss is not officially supported on Windows; use WSL2.

2. **Reddit API credentials** — https://www.reddit.com/prefs/apps
   - Create a "script" app
   - Note `client_id` and `client_secret`

## Setup

```bash
python -m venv ch4_env
source ch4_env/bin/activate
pip install -r requirements.txt
```

## Configuration

Edit `app.py` and set these near the bottom in `main()`:
- `REDDIT_CLIENT_ID`
- `REDDIT_CLIENT_SECRET`
- `REDDIT_USER_AGENT`
- `EXTENSION_PATH` (if binaries aren't in `.`)

## Run

```bash
# Verify sqlite-vss installation first
python -c "from app import verify_vss_installation; print(verify_vss_installation())"

# Run full pipeline
python app.py
```

## What it does

1. Fetches posts from Reddit via PRAW
2. Cleans/preprocesses text (markdown, URLs, reddit artifacts)
3. Generates embeddings with all-MiniLM-L6-v2
4. Stores in SQLite with VSS vector index
5. Performs semantic search with metadata filtering
6. Supports cross-subreddit analysis and similar-post discovery
