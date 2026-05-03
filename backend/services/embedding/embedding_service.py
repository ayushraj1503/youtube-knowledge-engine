# backend/services/embedding/embedding_service.py
"""
Embedding Service

Responsibilities:
  1. Load and cache sentence-transformers model (all-MiniLM-L6-v2)
  2. Generate embeddings in batches for efficiency
  3. Store embeddings + metadata in ChromaDB
  4. Provide existence checks to avoid re-embedding

Why all-MiniLM-L6-v2?
  - 384-dimensional dense vectors: compact but highly expressive for English
  - Runs on CPU in ~5ms per batch of 64 chunks — no GPU required
  - MIT licensed, widely benchmarked on semantic similarity tasks
  - Much faster than large models (e.g. mpnet-base-v2) with only ~5% quality loss

ChromaDB design decisions:
  - Persistent storage: embeddings survive restarts (no re-compute cost)
  - Metadata stored alongside vectors: enables pre-filter before ANN search
    e.g. WHERE channel_id = 'UCxxx' without loading all vectors
  - Collection = one namespace for all YouTube content (filtered by metadata)
  - Migration path: swap ChromaDB client for pgvector by implementing same
    interface in a PostgresVectorStore class

Caching strategy:
  - Disk cache (joblib.Memory): chunk_id → embedding
  - Avoids re-computing embeddings when reprocessing a video after a crash
  - Cache invalidated per chunk_id (hash of video_id + chunk_index)
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from backend.core.config import get_settings
from backend.core.logging_config import get_logger
from backend.models.schemas import TextChunk

logger = get_logger(__name__)
settings = get_settings()


class EmbeddingService:
    """
    Manages embedding generation and vector storage.
    Singleton-friendly: expensive model is loaded once.
    """

    _model: Optional[SentenceTransformer] = None  # class-level singleton

    def __init__(self):
        self._executor = ThreadPoolExecutor(
            max_workers=settings.EMBEDDING_CONCURRENCY,
            thread_name_prefix="embedder",
        )
        self._chroma_client = self._init_chroma()
        self._collection = self._get_or_create_collection()

    # ──────────────────────────────────────────────────────────────────────
    # INITIALISATION
    # ──────────────────────────────────────────────────────────────────────

    def _init_chroma(self) -> chromadb.PersistentClient:
        return chromadb.PersistentClient(
            path=str(settings.CHROMA_PERSIST_DIR),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

    def _get_or_create_collection(self):
        """
        ChromaDB collection with cosine similarity metric.
        cosine > euclidean for text embeddings because it's magnitude-invariant.
        """
        return self._chroma_client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load model once; reuse across all requests."""
        if EmbeddingService._model is None:
            logger.info("loading_embedding_model", model=settings.EMBEDDING_MODEL)
            EmbeddingService._model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info("embedding_model_loaded")
        return EmbeddingService._model

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────

    async def embed_and_store(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Generate embeddings for all chunks and store in ChromaDB.
        Runs in thread pool to avoid blocking the event loop.
        """
        if not chunks:
            return []

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._embed_and_store_sync,
            chunks,
        )

    def _embed_and_store_sync(self, chunks: List[TextChunk]) -> List[TextChunk]:
        model = self._get_model()
        texts = [c.text for c in chunks]

        # Batch encode — sentence-transformers handles batching internally
        logger.info("embedding_batch", count=len(texts))
        embeddings = model.encode(
            texts,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,   # L2 normalize for cosine via dot product
        )

        # Prepare ChromaDB upsert (idempotent — safe to re-run)
        ids = [c.chunk_id for c in chunks]
        documents = texts
        metadatas = [self._chunk_to_metadata(c) for c in chunks]
        embeddings_list = embeddings.tolist()

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings_list,
            metadatas=metadatas,
        )

        logger.info("embeddings_stored", count=len(chunks))
        return chunks

    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query string for similarity search."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._embed_query_sync,
            query,
        )

    def _embed_query_sync(self, query: str) -> List[float]:
        model = self._get_model()
        embedding = model.encode(
            [query],
            normalize_embeddings=True,
        )
        return embedding[0].tolist()

    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Vector similarity search in ChromaDB.
        Optional `where` dict enables metadata pre-filtering.

        ChromaDB returns results sorted by distance (lower = more similar).
        We convert to score = 1 - distance for intuitive 0→1 range.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._similarity_search_sync,
            query_embedding,
            top_k,
            where,
        )

    def _similarity_search_sync(
        self,
        query_embedding: List[float],
        top_k: int,
        where: Optional[Dict],
    ) -> List[Dict]:
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self._collection.count() or 1),
            "include": ["documents", "metadatas", "distances", "embeddings"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        chunks = []
        for i, (doc, meta, dist) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            chunks.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "score": float(1.0 - dist),  # cosine: distance → similarity
                    "rank": i,
                }
            )

        return chunks

    async def video_exists(self, video_id: str) -> bool:
        """Check if a video has already been ingested."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._video_exists_sync, video_id
        )

    def _video_exists_sync(self, video_id: str) -> bool:
        try:
            result = self._collection.get(
                where={"video_id": video_id},
                limit=1,
                include=[],
            )
            return len(result["ids"]) > 0
        except Exception:
            return False

    async def get_all_videos(self, page: int = 1, page_size: int = 20) -> Dict:
        """Return distinct videos stored in the collection."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._get_all_videos_sync, page, page_size
        )

    def _get_all_videos_sync(self, page: int, page_size: int) -> Dict:
        # ChromaDB doesn't support GROUP BY — we do deduplication in Python
        all_meta = self._collection.get(include=["metadatas"])["metadatas"]

        seen_ids = set()
        videos = []
        for meta in all_meta:
            vid_id = meta.get("video_id", "")
            if vid_id not in seen_ids:
                seen_ids.add(vid_id)
                videos.append(meta)

        total = len(videos)
        start = (page - 1) * page_size
        end = start + page_size
        return {
            "videos": videos[start:end],
            "total": total,
            "page": page,
            "page_size": page_size,
            "has_next": end < total,
        }

    # ──────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_to_metadata(chunk: TextChunk) -> Dict:
        """
        Flatten chunk into ChromaDB-compatible metadata dict.
        All values must be str | int | float | bool.
        """
        return {
            "video_id": chunk.video_id,
            "video_title": chunk.video_title,
            "channel_name": chunk.channel_name,
            "start_time": chunk.start_time,
            "end_time": chunk.end_time,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "frame_caption": chunk.frame_caption or "",
            **{k: str(v) for k, v in chunk.metadata.items()},
        }
