# hybrid-rag-core

A production-style Retrieval-Augmented Generation (RAG) pipeline built from scratch. Upload a PDF, ask questions, get answers grounded in your document.

Built to understand and demonstrate every step of the RAG pipeline — from PDF parsing to hybrid search to streaming LLM responses.

## Architecture

```
PDF Upload → pymupdf parsing → recursive chunking → TF-IDF embedding
    → Pinecone vector storage → hybrid retrieval (dense + BM25 + RRF)
    → LLM reranking → Groq/Llama 3.1 streaming response
```

```
┌─────────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  PDF Upload  │────→│  Chunker │────→│ Embedder │────→│ Pinecone │
└─────────────┘     └──────────┘     └──────────┘     └──────────┘
                                                            │
┌─────────────┐     ┌──────────┐     ┌──────────┐          │
│  Groq LLM   │←────│ Reranker │←────│ Retriever│←─────────┘
│  (streaming) │     │ (LLM)    │     │ (hybrid) │
└─────────────┘     └──────────┘     └──────────┘
```

## What Each Step Does

### 1. PDF Parsing (`pdf_parser.py`)
Extracts text from PDFs using **pymupdf** (fitz) — the best free PDF parser. Handles multi-column layouts, tables, and complex formatting better than PyPDF2.

### 2. Chunking (`chunker.py`)
**Recursive character splitting** — tries to split on natural boundaries in order: paragraphs → newlines → sentences → words. Keeps semantically related text together while respecting size limits.

### 3. Embedding (`embedder.py`)
**TF-IDF vectors** with fixed dimensions. Converts text to numerical vectors based on word importance. Each word's score = how often it appears in this chunk × how rare it is across all chunks. Designed to be swapped for neural embeddings (BGE, OpenAI, Sentence Transformers) when available.

### 4. Vector Storage (`vector_store.py`)
**Pinecone** — cloud vector database with HNSW indexing. Vectors are stored permanently and searched in O(log n) time instead of brute-force O(n). Falls back to in-memory search if Pinecone is unavailable.

### 5. Retrieval (`retriever.py`)
**Hybrid search** combining two approaches:
- **Dense search** (Pinecone): finds semantically similar chunks via vector cosine similarity
- **Sparse search** (BM25): finds keyword-matching chunks via term frequency scoring
- **Reciprocal Rank Fusion**: merges both ranked lists — chunks that score high in both get boosted

### 6. Reranking (`reranker.py`)
**LLM-based reranking** — after retrieving top 20 candidates, asks the LLM to score each chunk's relevance (0-10). Reorders by relevance score and returns top 5. A lightweight alternative to cross-encoder models.

### 7. Conversation Memory (`memory.py`)
**Sliding window + auto-summarization**:
- Keeps last 6 messages in full
- When history exceeds 12 messages, older messages are summarized by the LLM into 2-3 sentences
- Summary + recent messages are included in every prompt
- The LLM is stateless — memory is managed entirely in application code

### 8. Chat (`main.py`)
**Streaming SSE responses** via Groq (Llama 3.1 8B). The prompt includes conversation memory + retrieved context + the user's question. Tokens stream to the frontend as they're generated.

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| PDF parsing | pymupdf | Best free PDF parser, layout-aware |
| Chunking | Custom recursive splitter | Respects natural text boundaries |
| Embeddings | TF-IDF (swappable) | Zero dependencies, works offline |
| Vector DB | Pinecone (free tier) | Persistent, HNSW indexed, cloud |
| Sparse search | BM25 (custom) | Keyword matching, no dependencies |
| Fusion | Reciprocal Rank Fusion | Industry standard for hybrid search |
| Reranking | Groq LLM-based | No extra model needed |
| Chat LLM | Groq / Llama 3.1 8B | Free tier, fast inference |
| Backend | FastAPI + Python | Async, auto-docs, type-safe |
| Frontend | React + Vite + Tailwind | Fast dev, streaming SSE support |

## Setup

### Prerequisites
- Python 3.12+
- Node.js 18+
- [Groq API key](https://console.groq.com/keys) (free)
- [Pinecone API key](https://app.pinecone.io) (free tier)

### Backend
```bash
cd backend
pip install -r requirements.txt
```

Create `backend/.env`:
```
GROQ_API_KEY=gsk_your_key_here
PINECONE_API_KEY=pcsk_your_key_here
```

Run:
```bash
python -m uvicorn app.main:application --reload --port 8001
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/upload` | Upload a PDF — auto chunks, embeds, stores |
| POST | `/api/chat` | Chat with streaming SSE response |
| GET | `/api/health` | Health check |

## Design Decisions

**Why TF-IDF instead of neural embeddings?**
TF-IDF requires zero external dependencies — no PyTorch, no model downloads, no GPU. It works offline on any machine. The architecture supports swapping in neural embeddings (BGE, Sentence Transformers, Ollama) by changing one class. TF-IDF + BM25 hybrid search provides surprisingly good retrieval for most use cases.

**Why Pinecone over self-hosted?**
Free tier with 2GB storage, no infrastructure to manage, HNSW indexing out of the box. The code falls back to in-memory search if Pinecone is unavailable, so it works without any cloud dependency.

**Why LLM reranking instead of cross-encoder?**
Cross-encoders (ms-marco-MiniLM) require PyTorch (~800MB). LLM reranking uses the same Groq API we already have for chat — no new dependencies. Trade-off: ~500ms extra latency vs ~20ms for a cross-encoder.

**Why Groq over OpenAI?**
Free tier, fast inference (Llama 3.1 8B), no cost for experimentation. The LangChain abstraction makes it trivial to swap providers.

## RAG Pipeline Deep Dive

### Chunking Strategies Explored
| Strategy | How it works | Trade-off |
|----------|-------------|-----------|
| Fixed-size | Split every N chars | Simple but cuts mid-sentence |
| **Recursive** (used) | Split on paragraphs → sentences → words | Best general-purpose |
| Sentence-level | Split on sentence boundaries | Great for Q&A |
| Semantic | Group by embedding similarity | Best quality, slow |
| Token-based | Split by token count | Precise model budgets |
| Parent-child | Small chunks for search, large for context | Best precision + context |

### Embedding Approaches Explored
| Approach | Dims | Quality | Cost |
|----------|------|---------|------|
| **TF-IDF** (used) | Variable | Good for keywords | Free, zero deps |
| Sentence Transformers | 384 | Good semantic | Free, needs PyTorch |
| BGE-small | 384 | Better retrieval | Free, needs PyTorch |
| Ollama/nomic-embed | 768 | Good, local | Free, needs Ollama |
| OpenAI ada-002 | 1536 | Excellent | $0.02/1M tokens |

### Retrieval Techniques Explored
| Technique | What it catches | Used? |
|-----------|----------------|-------|
| Dense (TF-IDF cosine) | Word importance similarity | ✅ |
| Sparse (BM25) | Exact keyword matches | ✅ |
| **Hybrid (RRF fusion)** | Both semantic + keyword | ✅ |
| Reranking (LLM-based) | Fine-grained relevance | ✅ |
| Query rewriting | Expanded/clarified queries | Future |
| Multi-hop retrieval | Complex multi-part questions | Future |

## Future Improvements

### Near-term
- **Neural embeddings** — swap TF-IDF for BGE/Sentence Transformers for significantly better semantic retrieval
- **Cross-encoder reranking** — replace LLM reranker with ms-marco-MiniLM for 25x faster reranking
- **Query rewriting** — use LLM to expand vague queries before search
- **Multi-document support** — upload and search across multiple PDFs
- **Metadata filtering** — filter by page number, section, date

### Medium-term
- **RAGAS evaluation** — automated metrics (context precision, faithfulness, answer relevance) to measure and improve quality
- **Semantic chunking** — split by topic similarity instead of character count
- **Parent-child retrieval** — small chunks for precise search, large chunks for rich context
- **Persistent chat history** — store conversations in a database
- **Guardrails** — input/output filtering for prompt injection and PII

### Long-term
- **Knowledge graph** — extract entities and relationships for structured retrieval
- **Agentic RAG** — multi-step reasoning with tool use (LangGraph)
- **Fine-tuned embeddings** — domain-specific embedding model for better retrieval
- **Observability** — LangSmith/Langfuse tracing for debugging retrieval quality
- **A/B testing** — compare pipeline configurations with automated evaluation

## Project Structure

```
rag-dictionary/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app — upload + chat endpoints
│   │   ├── config.py        # Settings from .env
│   │   ├── ssl_fix.py       # Corporate network SSL patches
│   │   ├── pdf_parser.py    # pymupdf text extraction
│   │   ├── chunker.py       # Recursive character splitter
│   │   ├── embedder.py      # TF-IDF dense embeddings
│   │   ├── bm25.py          # BM25 sparse keyword search
│   │   ├── vector_store.py  # Pinecone storage + search
│   │   ├── retriever.py     # Hybrid search with RRF fusion
│   │   ├── reranker.py      # LLM-based reranking
│   │   └── memory.py        # Conversation memory + summarization
│   ├── requirements.txt
│   └── .env                 # API keys (not committed)
├── frontend/
│   ├── src/
│   │   └── App.tsx          # React chat UI with streaming
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

## License

MIT
