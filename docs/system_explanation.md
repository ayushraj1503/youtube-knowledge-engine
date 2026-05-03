# 📄 YouTube Knowledge Engine — System Explanation

> **Purpose of this document:** Help you understand, explain, and defend every design
> decision in this system during technical interviews. Written in plain English, not
> marketing language.

---

## Table of Contents

1. [What This System Does (in one paragraph)](#1-what-this-system-does)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Step-by-Step: How a Video Gets Ingested](#3-step-by-step-ingestion)
4. [Step-by-Step: How a Query Gets Answered](#4-step-by-step-query)
5. [Module-by-Module Breakdown](#5-module-breakdown)
6. [Why Each Tool Was Chosen](#6-tool-choices)
7. [How Hybrid Search Works](#7-hybrid-search)
8. [Key Design Decisions and Trade-offs](#8-design-decisions)
9. [How the System Scales](#9-scaling)
10. [Common Failure Points and Mitigations](#10-failure-handling)
11. [Interview Q&A Cheat Sheet](#11-interview-qa)

---

## 1. What This System Does

This system turns an entire YouTube channel into a searchable AI knowledge base.
You give it a channel URL. It downloads every video's transcript, breaks them into
chunks, turns each chunk into a mathematical vector (embedding), and stores everything
in a database. When you ask a question, the system finds the most relevant transcript
chunks across all videos, then uses an LLM (Groq) to write a coherent answer that
cites the exact video and timestamp where each piece of information came from.

The core idea is **Retrieval-Augmented Generation (RAG)**: instead of asking the LLM
to "remember" facts from YouTube videos (it can't — videos aren't in its training data),
we *retrieve* the relevant text first, then ask the LLM to *generate* an answer from
that retrieved context.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PATH                           │
│                                                                 │
│  YouTube Channel URL                                            │
│        │                                                        │
│        ▼                                                        │
│  yt-dlp ──────────────► Video Metadata (title, duration, etc.) │
│        │                                                        │
│        ▼                                                        │
│  youtube-transcript-api ► Raw transcript segments               │
│        │                                                        │
│        ▼                                                        │
│  Processing Pipeline:                                           │
│    1. Clean text (strip HTML, [Music] tags)                     │
│    2. Chunk into 512-word windows with 64-word overlap          │
│    3. FFmpeg → extract frames at 0.5 FPS                       │
│    4. BLIP → caption each frame                                 │
│    5. Append visual captions to chunk text                      │
│        │                                                        │
│        ▼                                                        │
│  Embedding Service:                                             │
│    all-MiniLM-L6-v2 → 384-dim float vector per chunk           │
│        │                                                        │
│        ▼                                                        │
│  ChromaDB ──────────────► Stored: vector + text + metadata      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        QUERY PATH                               │
│                                                                 │
│  User Question                                                  │
│        │                                                        │
│        ▼                                                        │
│  Query Rewriting (LLM) ──► keyword-enriched query              │
│        │                                                        │
│        ├──────────────────────────────────────┐                 │
│        ▼                                      ▼                 │
│  Vector Search                          BM25 Search             │
│  (ChromaDB ANN)                         (rank-bm25)             │
│        │                                      │                 │
│        └──────────────┬───────────────────────┘                 │
│                       ▼                                         │
│              Reciprocal Rank Fusion                             │
│              (merge ranked lists)                               │
│                       │                                         │
│                       ▼                                         │
│              Cross-Encoder Re-ranking                           │
│              (ms-marco-MiniLM)                                  │
│                       │                                         │
│                       ▼                                         │
│              Context Packing                                    │
│              (deduplicate, fit context window)                  │
│                       │                                         │
│                       ▼                                         │
│              Groq LLM (llama3-70b)                              │
│              Generate cited answer                              │
│                       │                                         │
│                       ▼                                         │
│              Response with timestamps + source links            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Step-by-Step: How a Video Gets Ingested

### Step 1: Channel Discovery (yt-dlp)

`yt-dlp` with `extract_flat=True` hits YouTube's API (no authentication needed
for public channels) and retrieves a flat list of all video IDs, titles, and
basic metadata. We use `extract_flat` specifically because it's fast — it doesn't
download the actual video info for each video, just the playlist/channel structure.

For a 500-video channel, this takes roughly 10-15 seconds.

### Step 2: Transcript Retrieval (youtube-transcript-api)

For each video ID, `youtube-transcript-api` fetches the auto-generated captions
from YouTube's `/timedtext` endpoint. These are free, no API key required,
and available for virtually every English video.

The transcript comes back as a list of segments:
```json
[
  {"text": "hello welcome to the channel", "start": 0.0, "duration": 3.2},
  {"text": "today we talk about neural nets", "start": 3.2, "duration": 2.8}
]
```

### Step 3: Text Cleaning

Auto-generated captions are noisy:
- YouTube injects `<c>` colour timing tags: `<c>Hello</c>`
- Sound annotations: `[Music]`, `[Applause]`, `[Laughter]`
- Inconsistent whitespace, truncated words

We use regex to strip all of this, then collapse whitespace. Short segments (< 3 chars)
get merged into adjacent ones to avoid embedding tiny meaningless strings.

### Step 4: Intelligent Chunking

This is one of the most important design decisions.

**Why not embed the whole transcript?**
- The embedding model (all-MiniLM-L6-v2) has a 256-token max input length.
  Even with truncation, a 30-minute video transcript (~4,500 words) won't fit.
- Even if it did, the vector would average over all topics discussed —
  retrieval would match "everything and nothing."

**Why chunking with overlap?**
- If a key insight spans two chunks, overlap ensures it appears in at least one
  complete chunk.
- Without overlap, a chunk boundary could cut through a sentence:
  "The best way to think about this is..." / "...like a rotating coordinate system"

**Our strategy:**
- 512-word windows (configurable)
- 64-word overlap (12.5%) between adjacent chunks
- Timestamps preserved: every chunk knows its `start_time` and `end_time`
- Each chunk gets a stable MD5 hash ID: `md5(video_id + chunk_index)`

### Step 5: Frame Extraction (FFmpeg) — Optional

FFmpeg reads the video stream URL (no download) at 0.5 FPS = one frame every 2 seconds.
For a 10-minute video, that's ~300 frames, capped at `MAX_FRAMES_PER_VIDEO=20`.

Why FFmpeg over OpenCV?
- FFmpeg handles AV1, VP9, H.264 without Python codec dependencies
- Can read from network URLs directly: `ffmpeg -i "https://..."` — no disk I/O
- Battle-tested, available as a system package in every Docker base image

### Step 6: Visual Captioning (BLIP)

For each chunk, we find the nearest extracted frame (by timestamp) and run it
through BLIP to get a short caption like: "a person drawing a diagram on a whiteboard".

We then append this to the chunk text:
```
[chunk text...] [Visual context: a person drawing a diagram on a whiteboard]
```

This means our embedding now captures both what was *said* and what was *shown* —
multi-modal retrieval without a multi-modal embedding model.

### Step 7: Embedding Generation (all-MiniLM-L6-v2)

The cleaned, enriched chunk text is passed through sentence-transformers in batches
of 64. Each chunk becomes a 384-dimensional float vector.

We use `normalize_embeddings=True` which L2-normalizes every vector. This means
cosine similarity = dot product, which ChromaDB can compute faster.

### Step 8: Storage (ChromaDB)

Each chunk is upserted into ChromaDB with:
- **id**: the chunk's MD5 hash (enables idempotent upserts — safe to re-run)
- **embedding**: 384-dim vector
- **document**: the chunk text
- **metadata**: video_id, title, channel, start_time, end_time, frame_caption, etc.

Metadata is stored flat (string/int/float values) because ChromaDB requires this.
All metadata is indexed for pre-filtering (e.g. `WHERE channel_id = 'UCxxx'`).

---

## 4. Step-by-Step: How a Query Gets Answered

### Step 1: Query Rewriting

User asks: *"how does he explain recursion"*

This is terrible for retrieval:
- "he" is ambiguous
- "explain" is a common word that matches everything
- No technical keywords

We send this to Groq with a tiny prompt:
> "Rewrite this query to maximize retrieval from a YouTube transcript database.
> Return ONLY the rewritten query."

Result: *"recursion computer science explanation base case recursive function"*

This dramatically improves recall for keyword-based search and precision for
vector search (because the embedding now matches technical vocabulary).

Cost: ~1 Groq API call, ~50 tokens, ~100ms additional latency. Worth it.

### Step 2: Dual Retrieval (Vector + BM25)

**Vector search:**
The rewritten query is embedded using the same model that embedded the chunks.
ChromaDB's HNSW index finds the top-K vectors by cosine similarity.
HNSW = Hierarchical Navigable Small World — a graph-based approximate nearest
neighbour algorithm. Fast, accurate, and scales well.

**BM25 search:**
All stored documents are loaded and scored with BM25-Okapi:
```
BM25(q, d) = Σ IDF(qᵢ) × (tf(qᵢ,d) × (k1+1)) / (tf(qᵢ,d) + k1×(1-b+b×|d|/avgdl))
```
BM25 rewards exact keyword matches with TF-IDF weighting.
It catches queries where specific technical terms are mentioned verbatim.

These two methods are complementary:
- Vector: handles paraphrasing, synonyms, semantic similarity
- BM25: handles exact terminology, named entities, acronyms

### Step 3: Reciprocal Rank Fusion (RRF)

We can't directly combine vector similarity scores (range: 0-1) with BM25 scores
(range: 0 to document-dependent maximum) — they're on completely different scales.

RRF solves this by using only *rank position*, not score values:
```
RRF_score(doc) = Σ weight_i / (k + rank_i(doc))
```
where k=60 (empirically optimal constant that smooths out rank differences).

A document ranked #1 in both lists gets `0.7/(60+1) + 0.3/(60+1) ≈ 0.016` —
much higher than a document ranked #5 in one and missing from the other.

We weight vector search at 0.7 and BM25 at 0.3 (configurable via `HYBRID_ALPHA`).
This reflects the general superiority of semantic search for open-ended questions,
while letting BM25 boost exact-match results.

### Step 4: Cross-Encoder Re-ranking

The bi-encoder (retrieval stage) is fast but approximate.
All-MiniLM encodes the query and each passage **independently** and compares the
resulting vectors. It can't model interactions between query and passage.

The cross-encoder (`ms-marco-MiniLM-L-6-v2`) takes *both* the query and passage
as a single input:
```
Input: [CLS] query [SEP] passage [SEP]
Output: relevance score (real number)
```
This is far more accurate because the model sees exactly how words in the query
relate to words in the passage. But it's 10-20× slower per passage.

Solution: **two-stage retrieval**
- Stage 1 (bi-encoder): retrieve 3×top_k candidates (~15 for top_k=5)
- Stage 2 (cross-encoder): score all 15, return top 5

We absorb the slow cross-encoder cost only over 15 passages, not the full corpus.

### Step 5: Context Window Optimization

We have 5 chunks, each ~512 words. That's ~2,560 words ≈ 3,400 tokens.
Llama3-70b has an 8,192 token context window. After reserving space for the
system prompt (~200 tokens) and answer (~500 tokens), we have ~7,500 tokens
for context.

Two additional optimizations:
1. **Deduplication**: if two chunks overlap in time from the same video (>70%),
   we drop the lower-scored one. Prevents the LLM from reading near-identical
   content twice, which causes repetition in answers.

2. **Lost-in-the-middle mitigation**: LLMs pay most attention to the beginning
   and end of their context window. Research (Liu et al., 2023) shows performance
   degrades for information placed in the middle. We move the highest-scored chunk
   to the *last* position so the LLM reads it last and prioritises it in its answer.

### Step 6: Answer Generation (Groq + llama3-70b-8192)

The final prompt is structured:
```
[SYSTEM] You are an expert assistant. Answer using only the provided sources.
         Cite sources as [SOURCE N]. Include timestamps.

[USER]   Context:
         [SOURCE 1] Video: "Intro to Neural Networks" | Timestamp: 04:23
         <chunk text>
         ---
         [SOURCE 2] Video: "Backprop Explained" | Timestamp: 12:07
         <chunk text>

         Question: How does backpropagation work?
```

We use `temperature=0.1` for factual, deterministic answers. Higher temperature
would give more creative but less accurate responses — wrong trade-off for a
knowledge retrieval system.

---

## 5. Module Breakdown

| Module | File | What it does |
|--------|------|-------------|
| **Config** | `core/config.py` | Pydantic Settings — typed, validated env vars. Single source of truth. |
| **Logging** | `core/logging_config.py` | structlog + rotating file handler. JSON for production, human-readable for dev. |
| **Schemas** | `models/schemas.py` | Pydantic v2 data contracts. All inter-service communication uses these types. |
| **ChannelIngester** | `services/ingestion/channel_ingester.py` | Asyncio orchestration. Fetches channel videos, dispatches to pipeline, tracks job state. |
| **ProcessingPipeline** | `services/processing/pipeline.py` | Clean → chunk → enrich → embed. The core transformation chain. |
| **FrameExtractor** | `services/processing/frame_extractor.py` | FFmpeg wrapper. Extracts keyframes from video stream URLs without downloading. |
| **VisualCaptioner** | `services/processing/visual_captioner.py` | BLIP wrapper. Lazy-loads model, captions frames for multimodal enrichment. |
| **EmbeddingService** | `services/embedding/embedding_service.py` | sentence-transformers + ChromaDB. Generates, caches, and stores vectors. |
| **RetrievalService** | `services/retrieval/retrieval_service.py` | Hybrid BM25 + vector search with RRF fusion and cross-encoder reranking. |
| **ResponseGenerator** | `services/generation/response_generator.py` | Query rewriting + context packing + Groq LLM answer generation. |
| **Routes** | `api/routes/*.py` | FastAPI endpoints: `/ingest/channel`, `/query`, `/videos`, `/health`. |
| **RateLimiter** | `middleware/rate_limiter.py` | Sliding window rate limiting per IP. Swap for Redis in multi-worker setup. |
| **Retry** | `utils/retry.py` | Async and sync exponential backoff decorators. Used on all external API calls. |

---

## 6. Why Each Tool Was Chosen

### yt-dlp (not pytube, not youtube-dl)
- `youtube-dl` is unmaintained since 2021; YouTube frequently changes its API
- `pytube` breaks often on age-restricted or certain channel formats
- `yt-dlp` is an actively maintained fork, battle-tested on millions of videos,
  with a plugin ecosystem and regular updates for YouTube API changes
- Critical for production: if YouTube changes something, yt-dlp is updated within days

### youtube-transcript-api (not Whisper)
- Whisper requires downloading the audio (large files), running speech recognition
  (slow), and produces lower accuracy than YouTube's own captions
- YouTube's auto-captions are already generated by a production ASR system and
  are free to access programmatically without authentication
- Trade-off: we lose videos with disabled captions (~5-10% of channels)
  In production, add Whisper as a fallback for these videos

### sentence-transformers / all-MiniLM-L6-v2 (not OpenAI text-embedding-ada-002)
- **Cost**: all-MiniLM is free. At 100,000 chunks, ada-002 costs ~$0.10.
  Sounds cheap, but at 1M chunks (a large channel archive) it's $1,000+.
- **Latency**: local inference on CPU is ~5ms/batch. No network round-trip.
- **Privacy**: embeddings never leave your infrastructure.
- **Quality**: 384-dim MiniLM scores 80%+ of ada-002 on MTEB benchmarks.
  The 20% quality gap is worth the 100× cost reduction for most use cases.
- Trade-off: if you need state-of-the-art embedding quality (e.g. for legal/medical
  search), upgrade to `all-mpnet-base-v2` (768-dim) or use OpenAI's latest embedding.

### ChromaDB (not Pinecone, not Weaviate, not pgvector)
- **Phase 1 simplicity**: ChromaDB runs as an embedded library — no separate
  database server to deploy, configure, or maintain. Single `pip install chromadb`.
- **Persistent**: data survives restarts via SQLite + parquet files on disk.
- **Metadata filtering**: WHERE clauses on stored metadata before vector search.
  This is crucial for filtering to a specific channel or video set.
- **Migration path**: the EmbeddingService interface is intentionally abstracted.
  Swap ChromaDB for pgvector by implementing the same methods on a PostgreSQL client.
- Trade-off: ChromaDB doesn't scale beyond ~500K vectors on a single node.
  For larger corpora, migrate to pgvector (Postgres) or Pinecone.

### Groq (not OpenAI GPT-4, not Anthropic Claude API)
- **Speed**: Groq runs llama3-70b at ~800 tokens/second on custom LPU hardware.
  OpenAI GPT-4 runs at ~20-50 tokens/second. For a RAG system, answer speed
  is UX-critical.
- **Cost**: Groq is cheaper per token than GPT-4 by ~10-30×.
- **Quality**: llama3-70b matches GPT-4 on most reasoning and summarisation tasks.
- Trade-off: Groq's rate limits are more restrictive than OpenAI's.
  For very high-traffic production use, add OpenAI as a fallback.

### BLIP (not GPT-4V, not LLaVA)
- BLIP runs entirely locally — no per-image API cost (critical when extracting
  20 frames × 500 videos = 10,000 captions)
- BLIP-base is 450MB — fits comfortably in a Docker container
- For higher quality captions, switch to BLIP-2 (better but needs 5GB+ VRAM)
- Trade-off: BLIP captions are shorter and less detailed than GPT-4V.
  For high-stakes visual retrieval, upgrade to a vision-language API.

### FastAPI (not Flask, not Django)
- **Async native**: FastAPI is built on Starlette/asyncio, which means ingestion
  jobs (network I/O heavy) run without blocking the event loop.
- **Auto-documentation**: Pydantic schemas → OpenAPI spec → Swagger UI for free.
- **Performance**: benchmarks consistently show FastAPI outperforming Flask by 2-3×
  on I/O-bound workloads, which is most of what this system does.
- **Type safety**: Pydantic validation catches bad inputs at the API boundary.

### FFmpeg (not OpenCV, not imageio)
- FFmpeg handles every video codec YouTube uses (H.264, VP9, AV1) without
  Python bindings or library conflicts
- Reading from a URL without downloading: `ffmpeg -i "https://stream.url" -vframes N`
  This is impossible with OpenCV's Python bindings without a custom build
- FFmpeg is a production-grade tool used in every major video platform

---

## 7. How Hybrid Search Works

The core insight is that no single search algorithm is universally best:

| Scenario | Vector wins | BM25 wins |
|----------|-------------|-----------|
| "What is the intuition behind attention?" | ✅ Semantic match | ❌ "intuition" not in transcript |
| "What does RLHF stand for?" | ❌ Vector is vague | ✅ Exact acronym match |
| "How to implement dropout regularisation?" | ✅ | ✅ Both work |
| "When did he mention PyTorch 2.0?" | ❌ Too specific | ✅ Keyword match |

Hybrid search combines both result lists using Reciprocal Rank Fusion (RRF).

**Why RRF specifically?**
Other fusion methods:
- **Weighted score combination**: requires normalising scores to the same range.
  BM25 scores are unbounded, cosine similarity is 0-1. Normalisation is fragile
  and dataset-dependent.
- **CombMNZ**: sums scores and multiplies by number of lists containing the document.
  Still requires score normalisation.
- **RRF**: uses only rank position. No normalisation needed. `1/(60 + rank)` means
  position 1 scores ~0.016, position 10 scores ~0.014 — the curve is deliberately
  flat to prevent one result from dominating just because it ranked #1 in one list.

**Parameter HYBRID_ALPHA=0.7:**
```
RRF_weight(doc) = 0.7 * (1/(60+rank_vector)) + 0.3 * (1/(60+rank_bm25))
```
0.7 biases toward semantic (vector) results. Adjust this based on your query types:
- Technical keyword-heavy queries → lower alpha (more BM25)
- Conceptual/open-ended queries → higher alpha (more vector)

---

## 8. Key Design Decisions and Trade-offs

### Decision 1: Async ingestion with job IDs
**Why:** Ingesting 500 videos takes 20-40 minutes. A synchronous HTTP request
would time out. The client needs a way to track progress.

**Implementation:** Fire-and-forget `asyncio.create_task()` returns instantly.
The job state is stored in a dict (`_jobs`) keyed by UUID. The client polls
`/ingest/status/{job_id}` every few seconds.

**Trade-off:** Job state is in-memory. If the server restarts, job history is
lost. For production: persist job state in Redis or a database.

### Decision 2: Word-based chunking (not token-based)
**Why:** Simple and predictable. Tokens are model-specific — what counts as 512
tokens in one model differs in another. Words are universal and easier to reason about.

**Trade-off:** Word count is only a proxy for token count (average ratio ~1.35).
A chunk of 512 words ≈ 690 tokens, which fits well within the 768-token limit of
all-MiniLM. For production with strict token budgets, use `tiktoken`.

### Decision 3: Upsert instead of insert
**Why:** Ingestion can fail mid-way. Re-running should not duplicate data.
ChromaDB's `upsert` uses chunk_id as the key — same chunk upserted twice
results in one stored record.

**Trade-off:** If a chunk's content changes (e.g. better transcript version),
an upsert correctly updates it. If you need to track versions, add a
`version` field to metadata.

### Decision 4: Class-level singleton for embedding model
**Why:** The sentence-transformers model is ~90MB loaded in memory. Loading it
per-request would OOM kill the server. Loading it per-instance would waste
memory in multi-worker setups.

**Implementation:** `EmbeddingService._model` is a class variable (not instance variable).
All instances share the same loaded model.

**Trade-off:** In a multi-process setup (e.g. `uvicorn --workers 4`), each worker
process has its own Python interpreter and thus its own model copy. That's 4×90MB = 360MB.
Acceptable. For GPU inference with large models, use a separate model server (e.g. TorchServe).

### Decision 5: In-memory rate limiter
**Why:** Simple, no external dependency, works fine for single-instance deployments.

**Trade-off:** In a multi-worker setup, each worker has independent rate limit
state — a client could send 100 requests/window to each of 4 workers = 400
effective requests. Fix: use Redis with `INCR`/`EXPIRE` for shared state.

### Decision 6: BM25 loads all documents into memory
**Why:** rank-bm25 is an in-memory library. For <50K chunks, it's fast enough.

**Trade-off:** At 500K+ chunks, loading all documents for every BM25 query is
too slow. Replace with Elasticsearch/OpenSearch or pgvector's `tsvector` full-text
search for large corpora.

---

## 9. How the System Scales

### Current limits (Phase 1 — ChromaDB embedded)
- **Vectors**: ChromaDB performs well up to ~500K vectors on a single node
- **BM25**: practical limit ~50K documents for in-memory loading
- **Concurrency**: 4 embedding workers, 4 ingestion workers

### Scaling to 1,000+ videos (~1M chunks)

**Step 1: Replace ChromaDB with pgvector**
```
CREATE EXTENSION vector;
ALTER TABLE chunks ADD COLUMN embedding vector(384);
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```
pgvector on PostgreSQL scales to tens of millions of vectors, supports complex
SQL filters, and handles concurrent writes properly.

**Step 2: Replace in-memory BM25 with Elasticsearch**
Elasticsearch's `match` query provides production-grade BM25 with:
- Inverted index (no full document scan)
- Distributed sharding
- Sub-millisecond keyword search at 100M+ documents

**Step 3: Separate embedding workers**
Move embedding generation to a dedicated worker fleet:
- Add a message queue (Redis Pub/Sub or Kafka)
- Ingestion service publishes chunks to the queue
- Embedding workers consume, generate vectors, write to pgvector
- This decouples ingestion throughput from embedding throughput

**Step 4: GPU inference for embeddings**
On a single NVIDIA A10G (24GB VRAM), all-MiniLM processes ~5,000 chunks/second
— 100× faster than CPU. For 1M chunks: CPU=55 hours → GPU=3.3 minutes.

**Step 5: Cache query embeddings**
The same question is often asked repeatedly. Cache query embedding → results
in Redis with TTL=1 hour. No LLM call needed for cached queries.

**Step 6: CDN for the Streamlit frontend**
Streamlit is single-threaded Python — it doesn't scale. For high-traffic:
replace with a React/Next.js frontend served via Cloudflare, with FastAPI
as the pure backend.

### Horizontal scaling diagram

```
                         ┌──────────────┐
Users ──► Load Balancer ─┤  FastAPI ×4  │
                         └──────┬───────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
        Redis Queue       pgvector DB       Redis Cache
              │                                   ▲
              ▼                                   │
     ┌────────────────┐                    Cache query
     │ Embedding      │                    results here
     │ Workers ×N     │
     │ (GPU optional) │
     └────────────────┘
```

---

## 10. Common Failure Points and How They Are Handled

### Failure 1: Video has no transcript
**Why it happens:** Channel owner disabled captions, video is non-English,
or YouTube hasn't generated auto-captions yet.

**Handling:**
- `TranscriptsDisabled` and `NoTranscriptFound` exceptions are caught
- Video is logged as failed but doesn't abort the job
- `videos_failed` counter is incremented in job status
- Future improvement: use Whisper as a fallback transcription engine

### Failure 2: Groq API rate limit
**Why it happens:** Groq's free tier has strict RPM (requests per minute) limits.
High query volume exhausts the rate limit.

**Handling:**
- `@sync_retry(max_attempts=3, delay=2.0, backoff=2.0)` on the Groq call
- Exponential backoff: first retry after 2s, second after 4s
- If all retries fail, a 500 error is returned to the user
- Production solution: implement token bucket client-side + circuit breaker

### Failure 3: FFmpeg stream URL expires
**Why it happens:** YouTube stream URLs are signed and expire after ~6 hours.
If frame extraction is slow, the URL may expire mid-extraction.

**Handling:**
- Frame extraction has a 120-second subprocess timeout
- If FFmpeg fails, `FrameExtractor.extract()` returns `[]`
- The pipeline continues without frame data — captions are skipped
- No crash, just degraded visual enrichment

### Failure 4: ChromaDB corruption
**Why it happens:** Power loss during a write, or running out of disk space.

**Handling:**
- ChromaDB uses SQLite + parquet; SQLite's WAL mode is crash-safe
- The `allow_reset=True` setting lets us reset the collection if needed
- Upsert idempotency means we can re-run ingestion to recover lost data
- Production: mount ChromaDB on a persistent volume with regular snapshots

### Failure 5: BLIP model fails to load
**Why it happens:** Not enough RAM/VRAM, network issue downloading from HuggingFace,
or incompatible CUDA version.

**Handling:**
- `VisualCaptioner._load_model()` catches all exceptions and sets `self._loaded = False`
- `ProcessingPipeline._enrich_with_visual_captions()` checks if captioner loaded
- If BLIP is unavailable, pipeline continues without visual enrichment
- `BLIP_ENABLED=false` in `.env` disables BLIP entirely as a kill switch

### Failure 6: Server restart loses in-memory job state
**Why it happens:** `_jobs` dict in `channel_ingester.py` is not persisted.

**Handling (current):** Job history is lost on restart. Any running jobs are abandoned.
**Production fix:** Persist job state to Redis or SQLite. Add job recovery logic
on startup that checks for `RUNNING` jobs and marks them as `FAILED`.

### Failure 7: Embedding model OOM on small machine
**Why it happens:** all-MiniLM requires ~200MB RAM loaded. With a large batch,
it can spike to 500MB+.

**Handling:**
- `EMBEDDING_BATCH_SIZE=64` limits memory per batch
- sentence-transformers uses numpy; no GPU memory fragmentation
- Reduce `EMBEDDING_BATCH_SIZE` in `.env` if OOM occurs (e.g. to 16)

---

## 11. Interview Q&A Cheat Sheet

**Q: What is RAG and why does it matter?**
> RAG (Retrieval-Augmented Generation) lets LLMs answer questions about content
> they've never been trained on. Instead of asking the model to remember facts,
> you retrieve relevant text at query time and give it to the model as context.
> This prevents hallucination and enables up-to-date answers without retraining.

**Q: Why not just fine-tune an LLM on the YouTube transcripts?**
> Fine-tuning is expensive (requires GPU infrastructure), slow (hours to days),
> and doesn't scale to new content — you'd need to retrain every time a new
> video is uploaded. RAG is cheaper, faster, and automatically handles new content
> as soon as it's indexed. Fine-tuning is better for teaching a model a *style*
> or *behaviour*, not for injecting factual knowledge.

**Q: How does chunking strategy affect retrieval quality?**
> Chunk size is a trade-off between specificity and completeness. Small chunks
> are retrieved with high precision but may miss surrounding context. Large chunks
> capture more context but the embedding averages over too many topics and retrieval
> precision drops. Overlap mitigates the boundary problem. Our 512-word/64-overlap
> choice is a common empirically-validated starting point. In production, you'd tune
> this by running retrieval evals on a sample of labelled queries.

**Q: What are the main failure modes of this system?**
> Three main ones: (1) Missing transcripts — handled by graceful fallback.
> (2) Retrieval misses — addressed by hybrid search and reranking. If the answer
> isn't in the top-K retrieved chunks, the LLM can't give a good answer.
> (3) LLM hallucination — mitigated by prompting the model to cite sources
> and only use provided context, plus low temperature.

**Q: How would you evaluate this system?**
> Two dimensions: retrieval quality and answer quality.
> For retrieval: create a test set of (query, expected_video_id) pairs.
> Compute Precision@K, Recall@K, and MRR. Track these as you tune chunk size,
> HYBRID_ALPHA, and whether re-ranking is enabled.
> For answer quality: use LLM-as-judge (ask a separate LLM to score answers 1-5
> on accuracy, completeness, and citation quality). Also track hallucination rate.

**Q: How would you handle 1 million videos?**
> Three main changes: (1) Replace ChromaDB with pgvector or Pinecone for the
> vector store. (2) Replace in-memory BM25 with Elasticsearch for keyword search.
> (3) Move embedding generation to a GPU worker fleet with a message queue.
> The FastAPI layer, retrieval logic, and LLM generation don't need to change.

**Q: Why Groq instead of OpenAI?**
> Speed and cost. Groq's custom LPU hardware runs llama3-70b at ~800 tokens/sec
> vs GPT-4's ~30 tokens/sec. For a RAG system where answer latency is UX-critical,
> that 25× speed difference is significant. Quality is comparable on summarisation
> and QA tasks. If you needed GPT-4-level reasoning (e.g. multi-step code generation),
> you'd reconsider.

**Q: What is the "lost in the middle" problem and how do you address it?**
> Research shows LLMs pay more attention to text at the beginning and end of their
> context window, and less to text in the middle. If we place the most relevant
> chunk in position 3 of 5, the model may underweight it. Our fix: reorder chunks
> so the highest-scored one is last in the context, where the model gives it
> maximum attention.

**Q: Is this system production-ready?**
> It's production-ready at small-to-medium scale (up to ~500 videos, low traffic).
> For true production at scale, you'd add: Redis for shared rate limiting and caching,
> pgvector for the vector store, Elasticsearch for BM25, job persistence, a proper
> authentication layer, a React frontend instead of Streamlit, monitoring (Prometheus
> + Grafana), and CI/CD pipelines. The architecture is explicitly designed to support
> these upgrades through its abstracted service interfaces.

---

*Document written to be interview-ready. Every design decision has a clear "why" and a known trade-off.*
