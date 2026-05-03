# backend/api/routes/query.py
"""Query / RAG endpoint."""

import time

from fastapi import APIRouter, HTTPException, Query, status

from backend.core.config import get_settings
from backend.core.logging_config import get_logger
from backend.models.schemas import (
    EvalRequest,
    EvalResponse,
    QueryRequest,
    QueryResponse,
    SearchMode,
)
from backend.services.generation.response_generator import ResponseGenerator
from backend.services.retrieval.retrieval_service import RetrievalService

router = APIRouter(prefix="/query", tags=["Query"])
logger = get_logger(__name__)
settings = get_settings()

_retriever = RetrievalService()
_generator = ResponseGenerator()


@router.post(
    "",
    response_model=QueryResponse,
    summary="Query the YouTube knowledge base",
)
async def query(request: QueryRequest):
    """
    Ask a question across all ingested YouTube content.

    **Flow:**
    1. (Optional) LLM rewrites query for better retrieval
    2. Hybrid BM25 + vector search retrieves top-K chunks
    3. Cross-encoder re-ranks candidates
    4. Groq LLM generates a cited answer

    **Returns:** Answer + timestamped source references
    """
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query cannot be empty",
        )

    try:
        # Step 1: Retrieve relevant chunks
        chunks = await _retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            mode=request.search_mode,
            rerank=request.rerank,
            channel_id=request.channel_id,
            video_ids=request.video_ids,
        )

        if not chunks:
            return QueryResponse(
                query=request.query,
                answer="No relevant content found. Please ingest a YouTube channel first.",
                sources=[],
                search_mode=request.search_mode,
                latency_ms=0.0,
            )

        # Step 2: Generate answer
        response = await _generator.generate(
            query=request.query,
            chunks=chunks,
            rewrite_query=request.rewrite_query,
            search_mode=request.search_mode,
        )

        return response

    except Exception as e:
        logger.error("query_error", query=request.query, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}",
        )


@router.post(
    "/eval",
    response_model=EvalResponse,
    summary="Evaluate retrieval quality",
)
async def evaluate_retrieval(request: EvalRequest):
    """
    Compute precision@k, recall@k, and MRR for a query against ground truth.
    Useful for A/B testing retrieval improvements.
    """
    chunks = await _retriever.retrieve(
        query=request.query,
        top_k=request.top_k,
        mode=SearchMode.HYBRID,
        rerank=True,
    )

    retrieved_ids = list({c.video_id for c in chunks})
    expected_set = set(request.expected_video_ids)
    retrieved_set = set(retrieved_ids)

    # Precision@K = relevant retrieved / total retrieved
    precision = len(expected_set & retrieved_set) / max(len(retrieved_set), 1)

    # Recall@K = relevant retrieved / total relevant
    recall = len(expected_set & retrieved_set) / max(len(expected_set), 1)

    # MRR = 1/rank of first relevant result
    mrr = 0.0
    for i, chunk in enumerate(chunks, 1):
        if chunk.video_id in expected_set:
            mrr = 1.0 / i
            break

    return EvalResponse(
        query=request.query,
        precision_at_k=round(precision, 4),
        recall_at_k=round(recall, 4),
        mrr=round(mrr, 4),
        retrieved_video_ids=retrieved_ids,
    )
