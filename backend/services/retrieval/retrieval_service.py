# backend/services/retrieval/retrieval_service.py
"""
Retrieval Service — Hybrid Search + Re-ranking

This is the most architecturally important module.
Retrieval quality directly determines answer quality.

How hybrid search works here:
  ┌─────────────┐      ┌──────────────┐
  │  BM25 (TF-  │      │  Dense       │
  │  IDF)       │      │  Vector      │
  │  Sparse     │      │  Semantic    │
  │  Retrieval  │      │  Retrieval   │
  └──────┬──────┘      └──────┬───────┘
         │                    │
         └─────────┬──────────┘
                   │  Reciprocal Rank Fusion (RRF)
                   │  or Weighted linear combination
                   ▼
           Merged candidate set
                   │
                   ▼  (optional)
           Cross-encoder Re-ranker
                   │
                   ▼
           Top-K final results

Why Reciprocal Rank Fusion (RRF)?
  - Score-free: BM25 and cosine similarity use completely different scales
  - RRF uses only rank position, so no normalisation needed
  - Empirically outperforms simple weighted score combination
  - Formula: RRF(d) = Σ 1/(k + rank_i(d)) where k=60 (constant)

Why re-ranking?
  - Bi-encoder (all-MiniLM) is fast but approximate
  - Cross-encoder (ms-marco-MiniLM-L-6-v2) reads query + passage together
    → much better relevance judgement, but too slow for first-pass retrieval
  - Two-stage: retrieve 3× candidates, re-rank, return top-K
"""

import asyncio
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from backend.core.config import get_settings
from backend.core.logging_config import get_logger
from backend.models.schemas import RetrievedChunk, SearchMode
from backend.services.embedding.embedding_service import EmbeddingService

logger = get_logger(__name__)
settings = get_settings()


def _make_timestamp_url(video_id: str, start_time: float) -> str:
    seconds = int(start_time)
    return f"https://www.youtube.com/watch?v={video_id}&t={seconds}s"


class RetrievalService:
    """
    Two-stage retrieval: first-pass candidates → optional re-ranking.
    """

    def __init__(self):
        self._embedding_service = EmbeddingService()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="retriever")
        self._reranker = None   # lazy loaded

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: SearchMode = SearchMode.HYBRID,
        rerank: bool = True,
        channel_id: Optional[str] = None,
        video_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """
        Main retrieval entry point.
        Runs first-pass retrieval then optional cross-encoder re-ranking.
        """
        # Metadata filter for ChromaDB pre-filtering
        where_filter = self._build_filter(channel_id, video_ids)

        # First-pass: retrieve more candidates than top_k for re-ranking
        candidate_k = top_k * 3 if rerank and settings.RERANKER_ENABLED else top_k

        if mode == SearchMode.HYBRID:
            candidates = await self._hybrid_retrieve(query, candidate_k, where_filter)
        elif mode == SearchMode.BM25:
            candidates = await self._bm25_retrieve(query, candidate_k, where_filter)
        else:
            candidates = await self._vector_retrieve(query, candidate_k, where_filter)

        # Second-pass: cross-encoder re-ranking
        if rerank and settings.RERANKER_ENABLED and len(candidates) > top_k:
            candidates = await self._rerank(query, candidates, top_k)
        else:
            candidates = candidates[:top_k]

        # Convert to response schema
        return [self._to_retrieved_chunk(c) for c in candidates]

    # ──────────────────────────────────────────────────────────────────────
    # VECTOR RETRIEVAL
    # ──────────────────────────────────────────────────────────────────────

    async def _vector_retrieve(
        self, query: str, top_k: int, where: Optional[Dict]
    ) -> List[Dict]:
        query_embedding = await self._embedding_service.embed_query(query)
        results = await self._embedding_service.similarity_search(
            query_embedding, top_k=top_k, where=where
        )
        return results

    # ──────────────────────────────────────────────────────────────────────
    # BM25 RETRIEVAL
    # ──────────────────────────────────────────────────────────────────────

    async def _bm25_retrieve(
        self, query: str, top_k: int, where: Optional[Dict]
    ) -> List[Dict]:
        """
        BM25 over all stored documents.
        Note: for large corpora (100K+ chunks) consider Elasticsearch instead.
        For <50K chunks, in-memory BM25 is fast and simple.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._bm25_retrieve_sync, query, top_k, where
        )

    def _bm25_retrieve_sync(
        self, query: str, top_k: int, where: Optional[Dict]
    ) -> List[Dict]:
        # Fetch all documents from ChromaDB
        collection = self._embedding_service._collection
        all_data = collection.get(
            where=where, include=["documents", "metadatas"]
        )

        docs = all_data.get("documents", [])
        metas = all_data.get("metadatas", [])

        if not docs:
            return []

        # Tokenise (simple whitespace — for production use NLTK word_tokenize)
        tokenized = [doc.lower().split() for doc in docs]
        bm25 = BM25Okapi(
            tokenized,
            k1=settings.BM25_K1,
            b=settings.BM25_B,
        )

        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]

        results = []
        max_score = max(scores[top_indices[0]], 1e-9)  # normalise BM25 score
        for i in top_indices:
            if scores[i] > 0:
                results.append(
                    {
                        "text": docs[i],
                        "metadata": metas[i],
                        "score": float(scores[i] / max_score),
                        "rank": len(results),
                    }
                )

        return results

    # ──────────────────────────────────────────────────────────────────────
    # HYBRID RETRIEVAL (RRF)
    # ──────────────────────────────────────────────────────────────────────

    async def _hybrid_retrieve(
        self, query: str, top_k: int, where: Optional[Dict]
    ) -> List[Dict]:
        """
        Parallel vector + BM25 retrieval, fused with Reciprocal Rank Fusion.
        """
        # Run both retrievals in parallel
        vector_results, bm25_results = await asyncio.gather(
            self._vector_retrieve(query, top_k, where),
            self._bm25_retrieve(query, top_k, where),
        )

        return self._reciprocal_rank_fusion(
            [vector_results, bm25_results],
            weights=[settings.HYBRID_ALPHA, 1.0 - settings.HYBRID_ALPHA],
            top_k=top_k,
        )

    def _reciprocal_rank_fusion(
        self,
        result_lists: List[List[Dict]],
        weights: List[float],
        top_k: int,
        k: int = 60,
    ) -> List[Dict]:
        """
        RRF score: Σ w_i / (k + rank_i(doc))

        Documents are identified by their text content (chunk text is unique per chunk_id).
        In production, use chunk_id for deduplication.
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_store: Dict[str, Dict] = {}

        for result_list, weight in zip(result_lists, weights):
            for rank, doc in enumerate(result_list):
                # Use text as key (chunk_id would be cleaner in production)
                key = doc["metadata"].get("video_id", "") + "_" + str(
                    doc["metadata"].get("chunk_index", rank)
                )
                rrf_scores[key] += weight / (k + rank + 1)
                doc_store[key] = doc

        # Sort by RRF score descending
        ranked_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

        results = []
        for i, key in enumerate(ranked_keys[:top_k]):
            doc = dict(doc_store[key])
            doc["score"] = rrf_scores[key]
            doc["rank"] = i
            results.append(doc)

        return results

    # ──────────────────────────────────────────────────────────────────────
    # CROSS-ENCODER RE-RANKING
    # ──────────────────────────────────────────────────────────────────────

    async def _rerank(
        self, query: str, candidates: List[Dict], top_k: int
    ) -> List[Dict]:
        """
        Cross-encoder re-ranking using sentence-transformers cross-encoder.
        The cross-encoder reads [query, passage] together → better relevance.
        ~10x slower than bi-encoder but only applied to top-K candidates.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._rerank_sync, query, candidates, top_k
        )

    def _rerank_sync(
        self, query: str, candidates: List[Dict], top_k: int
    ) -> List[Dict]:
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    max_length=512,
                )
                logger.info("reranker_loaded")
            except Exception as e:
                logger.warning("reranker_load_failed", error=str(e))
                return candidates[:top_k]

        pairs = [(query, c["text"]) for c in candidates]
        scores = self._reranker.predict(pairs)

        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    # ──────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _build_filter(
        self, channel_id: Optional[str], video_ids: Optional[List[str]]
    ) -> Optional[Dict]:
        """Build ChromaDB where-clause for metadata filtering."""
        conditions = []
        if channel_id:
            conditions.append({"channel_id": {"$eq": channel_id}})
        if video_ids:
            conditions.append({"video_id": {"$in": video_ids}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _to_retrieved_chunk(self, doc: Dict) -> RetrievedChunk:
        meta = doc.get("metadata", {})
        video_id = meta.get("video_id", "")
        start_time = float(meta.get("start_time", 0))

        return RetrievedChunk(
            chunk_id=meta.get("chunk_id", ""),
            video_id=video_id,
            video_title=meta.get("video_title", "Unknown"),
            channel_name=meta.get("channel_name", "Unknown"),
            text=doc.get("text", ""),
            start_time=start_time,
            end_time=float(meta.get("end_time", start_time + 30)),
            score=doc.get("score", 0.0),
            timestamp_url=_make_timestamp_url(video_id, start_time),
            frame_caption=meta.get("frame_caption") or None,
        )
