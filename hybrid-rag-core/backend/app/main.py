"""RAG Dictionary — Production-style RAG pipeline in a single app.

Upload PDF → chunk → embed → store in Pinecone → hybrid search → rerank → chat with Groq.
"""

import json
import logging
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import app.ssl_fix  # noqa: F401 — patches SSL for corporate networks
from app.config import settings
from app.pdf_parser import parse_pdf
from app.chunker import chunk_text
from app.embedder import TFIDFEmbedder
from app.bm25 import BM25
from app.vector_store import upsert as pinecone_upsert
from app.retriever import hybrid_search
from app.reranker import rerank
from app import memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

store: dict[str, Any] = {}

application = FastAPI(title="hybrid-rag-core", version="1.0.0")
application.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# ═══════════════════════════════════════════════
# UPLOAD
# ═══════════════════════════════════════════════

class UploadResponse(BaseModel):
    run_id: str
    pages: int
    chunks: int
    message: str
    storage: str


@application.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files supported")
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")

    text = parse_pdf(content)
    if not text.strip():
        raise HTTPException(400, "No text found in PDF")

    chunks = chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(400, "No chunks generated")

    run_id = str(uuid.uuid4())[:8]
    index_name = f"rag-{run_id}"
    embedder = TFIDFEmbedder(chunks, settings.EMBED_DIMS)
    bm25 = BM25(chunks)

    storage = "pinecone" if pinecone_upsert(index_name, chunks, embedder) else "in-memory"
    store[run_id] = {"chunks": chunks, "embedder": embedder, "bm25": bm25, "index_name": index_name}

    pages = text.count("\f") + 1
    logger.info("Uploaded %s: %d pages, %d chunks, storage=%s", file.filename, pages, len(chunks), storage)
    return UploadResponse(run_id=run_id, pages=pages, chunks=len(chunks),
        message=f"Processed {file.filename}: {len(chunks)} chunks", storage=storage)


# ═══════════════════════════════════════════════
# CHAT
# ═══════════════════════════════════════════════

class ChatRequest(BaseModel):
    run_id: str
    message: str


@application.post("/api/chat")
async def chat(body: ChatRequest):
    data = store.get(body.run_id)
    if not data:
        raise HTTPException(404, "Upload a PDF first.")

    # Summarize old messages if needed
    await memory.maybe_summarize(body.run_id)

    # Retrieve → Rerank
    retrieved = hybrid_search(body.message, data["chunks"], data["embedder"], data["bm25"], data["index_name"])
    reranked = await rerank(body.message, retrieved)

    context = "\n\n".join(f"[{i+1}] {chunk}" for i, (chunk, _) in enumerate(reranked))
    mem = memory.build_memory_prompt(body.run_id)

    prompt = f"""You are a helpful assistant answering questions about an uploaded document.
Use the retrieved context to answer. If the context doesn't contain the answer, say so.
Use conversation history to understand follow-up questions.

{mem}

[Retrieved context]
{context}

User: {body.message}

Answer:"""

    sources = [{"text": chunk, "score": round(score, 4)} for chunk, score in reranked]

    async def stream():
        yield f"data: {json.dumps({'type': 'sources', 'chunks': sources})}\n\n"
        full_response = ""
        if settings.GROQ_API_KEY:
            try:
                from langchain_groq import ChatGroq
                from langchain_core.messages import HumanMessage
                http_client = httpx.AsyncClient(verify=False)
                llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=settings.GROQ_API_KEY, streaming=True, temperature=0.7, http_async_client=http_client)
                async for chunk in llm.astream([HumanMessage(content=prompt)]):
                    if hasattr(chunk, "content") and chunk.content:
                        full_response += chunk.content
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            except Exception as e:
                logger.error("Groq error: %s", e)
                full_response = f"Error: {e}"
                yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
        else:
            full_response = "Set GROQ_API_KEY in .env"
            yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        memory.add_message(body.run_id, "user", body.message)
        memory.add_message(body.run_id, "assistant", full_response)

    return StreamingResponse(stream(), media_type="text/event-stream")


@application.get("/api/health")
async def health():
    from app.vector_store import pc
    return {"status": "ok", "runs": len(store), "pinecone": pc is not None}
