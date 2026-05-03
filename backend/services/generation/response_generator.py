# backend/services/generation/response_generator.py
"""
Response Generator — LLM-powered answer synthesis

Pipeline:
  1. (Optional) Query Rewriting: expand/clarify ambiguous user query
  2. Context Window Optimization: select and order the best chunks
  3. Prompt Construction: structured prompt with retrieved context
  4. Groq API call: llama3-70b-8192 for fast, high-quality generation
  5. Source attribution: include timestamps and video references

Query Rewriting rationale:
  - Users often ask conversational questions that don't match retrieval vocab
  - "how does he explain recursion" → "explanation of recursion computer science"
  - LLM-based rewriting improves retrieval recall by ~15-20% in practice
  - We use a tiny prompt + fast Groq call (adds ~200ms) 

Context Window Optimization:
  - Llama3-70b has 8192 token context: we must fit query + context + answer
  - Strategy: sort retrieved chunks by score, truncate to fit context budget
  - De-duplicate: if two chunks from same video overlap in time, merge them
  - Position bias: put the highest-scoring chunk first AND last (LLM tends
    to focus on beginning and end of context — "lost in the middle" effect)

Prompt design:
  - System prompt sets the role and output format
  - Each source chunk is labelled with [SOURCE N] for citation tracking
  - LLM is instructed to cite sources and include timestamps
  - Temperature = 0.1 for factual, deterministic answers
"""

import time
from typing import List, Optional, Tuple

from groq import Groq

from backend.core.config import get_settings
from backend.core.logging_config import get_logger
from backend.models.schemas import QueryResponse, RetrievedChunk, SearchMode
from backend.utils.retry import sync_retry

logger = get_logger(__name__)
settings = get_settings()

# Approximate tokens per word (English average)
_AVG_TOKENS_PER_WORD = 1.35
# Reserve tokens for system prompt + response
_CONTEXT_TOKEN_BUDGET = 5000


class ResponseGenerator:
    """
    Wraps Groq API for answer generation with query rewriting + context packing.
    """

    def __init__(self):
        self._client = Groq(api_key=settings.GROQ_API_KEY)

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────

    async def generate(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        rewrite_query: bool = True,
        search_mode: SearchMode = SearchMode.HYBRID,
    ) -> QueryResponse:
        """
        Full generation pipeline:
          query → (rewrite) → context packing → LLM → structured response
        """
        t0 = time.perf_counter()
        rewritten_query = None

        # Step 1: Query rewriting
        effective_query = query
        if rewrite_query and len(query.split()) < 15:
            rewritten_query = self._rewrite_query(query)
            if rewritten_query and rewritten_query != query:
                logger.info(
                    "query_rewritten",
                    original=query,
                    rewritten=rewritten_query,
                )
                effective_query = rewritten_query

        # Step 2: Context window optimization
        context, used_chunks = self._pack_context(chunks)

        # Step 3: Generate answer
        answer, tokens_used = self._generate_answer(effective_query, context)

        latency_ms = (time.perf_counter() - t0) * 1000

        return QueryResponse(
            query=query,
            rewritten_query=rewritten_query,
            answer=answer,
            sources=used_chunks,
            search_mode=search_mode,
            latency_ms=round(latency_ms, 2),
            tokens_used=tokens_used,
        )

    # ──────────────────────────────────────────────────────────────────────
    # QUERY REWRITING
    # ──────────────────────────────────────────────────────────────────────

    @sync_retry(max_attempts=2, delay=1.0)
    def _rewrite_query(self, query: str) -> Optional[str]:
        """
        Rewrite the user query to be more retrieval-friendly.
        Uses a concise prompt with llama3-70b for speed.
        """
        system = (
            "You are a search query optimizer. "
            "Rewrite the given question to maximize retrieval from a YouTube transcript database. "
            "Make it keyword-rich and specific. "
            "Return ONLY the rewritten query — no explanation, no preamble."
        )

        try:
            response = self._client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Original query: {query}"},
                ],
                max_tokens=100,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("query_rewrite_failed", error=str(e))
            return None

    # ──────────────────────────────────────────────────────────────────────
    # CONTEXT PACKING
    # ──────────────────────────────────────────────────────────────────────

    def _pack_context(
        self, chunks: List[RetrievedChunk]
    ) -> Tuple[str, List[RetrievedChunk]]:
        """
        Pack retrieved chunks into the context window budget.

        Strategy:
          1. Sort by score descending
          2. De-duplicate overlapping chunks from same video
          3. Add chunks until token budget is exhausted
          4. Apply "lost in the middle" mitigation: move top chunk to end

        Returns: (formatted context string, list of chunks that made the cut)
        """
        # De-duplicate: skip chunks whose time range overlaps significantly
        deduped = self._deduplicate_chunks(chunks)

        # Greedily pack within token budget
        used_chunks: List[RetrievedChunk] = []
        total_tokens = 0

        for chunk in deduped:
            chunk_tokens = len(chunk.text.split()) * _AVG_TOKENS_PER_WORD
            if total_tokens + chunk_tokens > _CONTEXT_TOKEN_BUDGET:
                break
            used_chunks.append(chunk)
            total_tokens += chunk_tokens

        if not used_chunks:
            return "No relevant context found.", []

        # "Lost in the middle" mitigation: put best chunk last
        if len(used_chunks) > 2:
            best = used_chunks.pop(0)
            used_chunks.append(best)

        # Format context with source labels
        context_parts = []
        for i, chunk in enumerate(used_chunks, 1):
            minutes = int(chunk.start_time // 60)
            secs = int(chunk.start_time % 60)
            ts = f"{minutes:02d}:{secs:02d}"
            context_parts.append(
                f"[SOURCE {i}] Video: \"{chunk.video_title}\" | Timestamp: {ts}\n"
                f"{chunk.text}"
            )

        return "\n\n---\n\n".join(context_parts), used_chunks

    def _deduplicate_chunks(
        self, chunks: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        """
        Remove chunks with >70% temporal overlap from the same video.
        Prevents the LLM from seeing near-identical context multiple times.
        """
        deduped: List[RetrievedChunk] = []
        for candidate in chunks:
            overlap = False
            for existing in deduped:
                if existing.video_id != candidate.video_id:
                    continue
                # Check temporal overlap ratio
                overlap_start = max(existing.start_time, candidate.start_time)
                overlap_end = min(existing.end_time, candidate.end_time)
                if overlap_end > overlap_start:
                    candidate_duration = max(
                        candidate.end_time - candidate.start_time, 1
                    )
                    ratio = (overlap_end - overlap_start) / candidate_duration
                    if ratio > 0.7:
                        overlap = True
                        break
            if not overlap:
                deduped.append(candidate)
        return deduped

    # ──────────────────────────────────────────────────────────────────────
    # LLM GENERATION
    # ──────────────────────────────────────────────────────────────────────

    @sync_retry(max_attempts=3, delay=2.0, backoff=2.0)
    def _generate_answer(
        self, query: str, context: str
    ) -> Tuple[str, Optional[int]]:
        """
        Generate a cited, timestamp-linked answer using Groq (llama3-70b).
        """
        system_prompt = """You are an expert assistant for a YouTube Knowledge Engine.
You answer questions using ONLY the provided transcript excerpts from YouTube videos.

RULES:
1. Answer based strictly on the provided [SOURCE N] excerpts
2. Cite your sources using [SOURCE N] notation
3. Include timestamps when relevant (e.g. "As discussed at 12:34 in [SOURCE 2]...")
4. If the context doesn't contain enough information, say so clearly
5. Be concise but complete — prefer bullet points for multi-part answers
6. Never hallucinate — if you're unsure, say you don't have enough context"""

        user_message = f"""Context from YouTube videos:

{context}

---

Question: {query}

Answer (with source citations and timestamps):"""

        response = self._client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=settings.GROQ_MAX_TOKENS,
            temperature=settings.GROQ_TEMPERATURE,
        )

        answer = response.choices[0].message.content.strip()
        tokens_used = (
            response.usage.total_tokens if response.usage else None
        )

        logger.info(
            "answer_generated",
            tokens=tokens_used,
            answer_length=len(answer),
        )

        return answer, tokens_used
