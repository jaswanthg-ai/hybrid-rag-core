"""Pinecone vector store — persistent dense search with HNSW indexing."""

import logging
from typing import Any

from app.config import settings
from app.embedder import TFIDFEmbedder

logger = logging.getLogger(__name__)

pc = None
try:
    from pinecone import Pinecone

    if settings.PINECONE_API_KEY:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        logger.info("Pinecone connected")
except Exception as e:
    logger.warning("Pinecone not available: %s", e)


def upsert(index_name: str, chunks: list[str], embedder: TFIDFEmbedder) -> bool:
    """Store chunks + vectors in Pinecone. Returns True on success."""
    if not pc:
        return False
    try:
        existing = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing:
            from pinecone import ServerlessSpec

            pc.create_index(
                name=index_name,
                dimension=settings.EMBED_DIMS,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info("Created Pinecone index: %s", index_name)

        index = pc.Index(index_name)
        vectors = embedder.embed_all(chunks)
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = [
                {
                    "id": f"chunk-{j}",
                    "values": vectors[j],
                    "metadata": {"text": chunks[j][:40000], "chunk_index": j},
                }
                for j in range(i, min(i + batch_size, len(chunks)))
            ]
            index.upsert(vectors=batch)
        logger.info("Upserted %d vectors to '%s'", len(chunks), index_name)
        return True
    except Exception as e:
        logger.error("Pinecone upsert failed: %s", e)
        return False


def search(index_name: str, query_vector: list[float], top_k: int = 20) -> list[dict]:
    """Dense search via Pinecone. Returns [{text, score, chunk_index}]."""
    if not pc:
        return []
    try:
        index = pc.Index(index_name)
        results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        return [
            {"text": m.metadata.get("text", ""), "score": m.score, "chunk_index": m.metadata.get("chunk_index", 0)}
            for m in results.matches
        ]
    except Exception as e:
        logger.error("Pinecone search error: %s", e)
        return []
