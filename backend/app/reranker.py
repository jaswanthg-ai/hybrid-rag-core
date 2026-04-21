"""LLM-based reranker using Groq.

Scores each retrieved chunk's relevance to the query using the LLM,
then returns only the top-k most relevant chunks. This is a lightweight
alternative to cross-encoder models (ms-marco-MiniLM, bge-reranker).
"""

import json
import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


async def rerank(query: str, chunks: list[tuple[str, float]], top_k: int | None = None) -> list[tuple[str, float]]:
    """Rerank chunks by relevance using Groq LLM. Returns [(text, score)]."""
    top_k = top_k or settings.RERANK_TOP_K

    if not settings.GROQ_API_KEY or len(chunks) <= top_k:
        return chunks[:top_k]

    # Build scoring prompt
    chunk_list = "\n".join(f"[{i}] {text[:300]}" for i, (text, _) in enumerate(chunks))
    prompt = f"""Score each chunk's relevance to the question on a scale of 0-10.
Return ONLY a JSON array of scores, e.g. [8, 3, 9, 1, 5]

Question: {query}

Chunks:
{chunk_list}

Scores (JSON array):"""

    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage

        http_client = httpx.AsyncClient(verify=False)
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=settings.GROQ_API_KEY,
            temperature=0,
            http_async_client=http_client,
        )
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        # Parse scores from LLM response
        text = resp.content.strip()
        # Extract JSON array from response
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            scores = json.loads(text[start:end])
            if len(scores) == len(chunks):
                scored = [(chunks[i][0], float(scores[i])) for i in range(len(chunks))]
                scored.sort(key=lambda x: x[1], reverse=True)
                logger.info("Reranked %d chunks, top score: %.1f", len(chunks), scored[0][1])
                return scored[:top_k]

        logger.warning("Reranker returned unexpected format: %s", text[:100])
    except Exception as e:
        logger.warning("Reranker failed: %s — using original order", e)

    return chunks[:top_k]
