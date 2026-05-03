# 🎬 YouTube Knowledge Engine

> **Production-grade RAG system** — Turn any YouTube channel into a searchable AI knowledge base with timestamped answers.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-FF6B35?style=flat)](https://www.trychroma.com)
[![Groq](https://img.shields.io/badge/Groq-llama3--70b-F55036?style=flat)](https://groq.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat&logo=docker)](https://docker.com)

---

## 👤 About This Project

Built this to solve a real problem I faced — 
watching 10+ hours of ML lecture videos and not 
being able to find specific explanations later.

Started with basic transcript search, realized 
keyword search missed semantic meaning, added 
vector embeddings. Then noticed exact terms like 
"RLHF" weren't matching semantically so added BM25. 
Kept iterating from there.

**Biggest challenge:** YouTube's transcript API 
changed between versions — had to debug the 
new fetch() method vs old get_transcript().

**What I learned:** The "lost in the middle" 
problem in LLMs was surprising — moving the 
best chunk to last position actually improved 
answer quality noticeably.


## What It Does

1. **Ingest** an entire YouTube channel (hundreds of videos)
2. **Extract** transcripts + visual frames + BLIP captions
3. **Index** everything with hybrid BM25 + vector embeddings in ChromaDB
4. **Answer** questions with timestamped citations to exact video moments

---

## Project Structure

```
youtube-knowledge-engine/
│
├── backend/                        # FastAPI application
│   ├── main.py                     # App entrypoint, middleware, routers
│   ├── core/
│   │   ├── config.py               # Pydantic Settings (all env vars)
│   │   └── logging_config.py       # structlog + rotating file handler
│   ├── models/
│   │   └── schemas.py              # Pydantic v2 data contracts
│   ├── services/
│   │   ├── ingestion/
│   │   │   └── channel_ingester.py # yt-dlp + async orchestration
│   │   ├── processing/
│   │   │   ├── pipeline.py         # Clean → chunk → embed pipeline
│   │   │   ├── frame_extractor.py  # FFmpeg frame extraction
│   │   │   └── visual_captioner.py # BLIP image captioning
│   │   ├── embedding/
│   │   │   └── embedding_service.py# all-MiniLM + ChromaDB
│   │   ├── retrieval/
│   │   │   └── retrieval_service.py# Hybrid BM25+vector + reranking
│   │   └── generation/
│   │       └── response_generator.py # Query rewriting + Groq LLM
│   ├── api/routes/
│   │   ├── ingest.py               # POST /api/v1/ingest/channel
│   │   ├── query.py                # POST /api/v1/query
│   │   ├── videos.py               # GET  /api/v1/videos
│   │   └── health.py               # GET  /health
│   ├── middleware/
│   │   └── rate_limiter.py         # Per-IP sliding window limiter
│   └── utils/
│       └── retry.py                # Async + sync retry decorators
│
├── frontend/
│   └── app.py                      # Streamlit UI
│
├── docs/
│   └── system_explanation.md       # 📄 Full technical deep-dive
│
├── tests/
│   ├── unit/test_pipeline.py       # Unit tests: chunking, cleaning
│   └── integration/test_api.py     # HTTP endpoint tests
│
├── scripts/
│   ├── setup.sh                    # One-command setup
│   └── quickstart_demo.py          # Demo: ingest + query
│
├── infrastructure/
│   ├── docker/
│   │   ├── Dockerfile.backend
│   │   └── Dockerfile.frontend
│   └── nginx/                      # (for production SSL termination)
│
├── data/                           # Runtime data (gitignored)
│   ├── chroma_db/
│   ├── logs/
│   └── cache/
│
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## Quick Start

### Option A: Local (Recommended for Development)

```bash
# 1. Clone and setup
git clone <repo-url>
cd youtube-knowledge-engine
chmod +x scripts/setup.sh && ./scripts/setup.sh

# 2. Add your Groq API key
# Get one free at https://console.groq.com
nano .env   # set GROQ_API_KEY=gsk_...

# 3. Activate venv and start backend
source venv/bin/activate
uvicorn backend.main:app --reload --port 8000

# 4. Start frontend (new terminal)
source venv/bin/activate
streamlit run frontend/app.py
```

Open:
- **Frontend**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

### Option B: Docker (Recommended for Production)

```bash
# 1. Configure environment
cp .env.example .env
nano .env   # set GROQ_API_KEY

# 2. Build and run
docker-compose up --build

# 3. Access
# Frontend: http://localhost:8501
# API:      http://localhost:8000/docs
```

---

## API Endpoints

### `POST /api/v1/ingest/channel`
Start async ingestion of a YouTube channel.
```json
{
  "channel_url": "https://www.youtube.com/@3blue1brown",
  "max_videos": 50,
  "force_reingest": false,
  "extract_frames": true
}
```
Returns `202 Accepted` with a `job_id` immediately.

### `GET /api/v1/ingest/status/{job_id}`
Poll ingestion progress.
```json
{
  "status": "running",
  "videos_total": 50,
  "videos_processed": 23,
  "chunks_created": 1847,
  "progress_percent": 46.0
}
```

### `POST /api/v1/query`
Ask a question across all ingested content.
```json
{
  "query": "How does backpropagation work?",
  "top_k": 5,
  "search_mode": "hybrid",
  "rerank": true,
  "rewrite_query": true
}
```
Returns an answer with timestamped source citations.

### `GET /api/v1/videos?page=1&page_size=20`
Browse indexed videos with pagination.

### `GET /health`
System health check for all components.

---

## Example Queries

After ingesting a machine learning channel:
```
"What is the difference between supervised and unsupervised learning?"
"How does the attention mechanism work in transformers?"
"What tools does he recommend for beginners getting into ML?"
"Explain gradient descent with a real-world analogy"
"When did he first mention PyTorch 2.0?"
"What projects should I build to learn deep learning?"
"How can I improve model accuracy without collecting more data?"
```

After ingesting a business/finance channel:
```
"What does he say about the current state of the housing market?"
"What investment strategies does he recommend for beginners?"
"How should I think about risk tolerance in my 30s?"
```

---

## Configuration Reference

Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | **required** | Get free at console.groq.com |
| `CHUNK_SIZE` | `512` | Words per chunk (try 256-1024) |
| `CHUNK_OVERLAP` | `64` | Overlap words between chunks |
| `HYBRID_ALPHA` | `0.7` | Vector vs BM25 weight (1.0=pure vector) |
| `TOP_K` | `5` | Retrieved chunks per query |
| `RERANKER_ENABLED` | `true` | Cross-encoder re-ranking |
| `BLIP_ENABLED` | `true` | Visual frame captioning |
| `FRAME_EXTRACTION_FPS` | `0.5` | Frames per second to extract |
| `YTDLP_MAX_VIDEOS` | `500` | Cap on videos per channel |
| `INGESTION_CONCURRENCY` | `4` | Parallel video processing |

---

## Running Tests

```bash
# Unit tests only (no backend required)
pytest tests/unit/ -v

# Integration tests (requires running backend)
uvicorn backend.main:app --port 8000 &
pytest tests/integration/ -v

# All tests with coverage
pytest --cov=backend --cov-report=html
```

---

## Architecture Deep-Dive

See **[docs/system_explanation.md](docs/system_explanation.md)** for:
- Step-by-step data flow explanation
- Why each tool was chosen
- How hybrid search and RRF work
- Scaling strategy for 1000+ videos
- Common failure points and mitigations
- Interview Q&A cheat sheet

---

## Scaling Strategy

| Scale | Vector DB | BM25 | Embedding |
|-------|-----------|------|-----------|
| < 500 videos | ChromaDB (current) | rank-bm25 in-memory | CPU, sentence-transformers |
| 500-5K videos | pgvector on PostgreSQL | Elasticsearch | GPU worker fleet |
| 5K+ videos | Pinecone / Weaviate | Elasticsearch cluster | Dedicated embedding API |

The service interfaces are designed for this migration — swap implementations
without changing the API layer.

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **API** | FastAPI + Uvicorn | Async REST API, auto-docs |
| **LLM** | Groq (llama3-70b) | Fast answer generation |
| **Embeddings** | all-MiniLM-L6-v2 | Semantic vector generation |
| **Vector DB** | ChromaDB | ANN search + metadata filtering |
| **BM25** | rank-bm25 | Keyword search |
| **Re-ranking** | ms-marco-MiniLM | Cross-encoder relevance scoring |
| **Video scraping** | yt-dlp | Channel metadata + stream URLs |
| **Transcripts** | youtube-transcript-api | Caption extraction |
| **Frame extraction** | FFmpeg | Keyframe extraction from streams |
| **Vision model** | BLIP (HuggingFace) | Local image captioning |
| **Frontend** | Streamlit | Interactive search UI |
| **Logging** | structlog | Structured JSON logging |
| **Deployment** | Docker + docker-compose | Container orchestration |

---

*Built as a production-grade RAG system demonstrating real engineering practices.*
