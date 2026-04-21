"""Hybrid retrieval: Pinecone dense + BM25 sparse + Reciprocal Rank Fusion."""

from app.bm25 import BM25
from app.config import settings
from app.embedder import TFIDFEmbedder
from app.vector_store import search as pinecone_search


def hybrid_search(
    query: str,
    chunks: list[str],
    embedder: TFIDFEmbedder,
    bm25: BM25,
    index_name: str,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Hybrid retrieval with RRF fusion. Returns [(chunk_text, score)]."""
    top_k = top_k or settings.RETRIEVE_TOP_K
    query_vec = embedder.embed(query)

    # Dense: Pinecone HNSW search
    dense_results = pinecone_search(index_name, query_vec, top_k=top_k)
    dense_ranked = {r["chunk_index"]: rank for rank, r in enumerate(dense_results)}

    # Sparse: BM25 keyword search
    bm25_scores = bm25.score(query)
    sparse_sorted = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
    sparse_ranked = {idx: rank for rank, (idx, _) in enumerate(sparse_sorted)}

    # Reciprocal Rank Fusion
    rrf_k = 60
    fused: dict[int, float] = {}
    for idx in set(dense_ranked) | set(sparse_ranked):
        d = dense_ranked.get(idx, len(chunks))
        s = sparse_ranked.get(idx, len(chunks))
        fused[idx] = 1.0 / (rrf_k + d) + 1.0 / (rrf_k + s)

    top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(chunks[idx], score) for idx, score in top if idx < len(chunks)]
