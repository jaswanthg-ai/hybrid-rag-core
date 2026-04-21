"""Conversation memory with sliding window + auto-summarization.

Keeps recent messages in full and summarizes older ones to stay within
token budgets. The LLM is stateless — memory is managed entirely here.
"""

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# In-memory stores (per run_id)
history: dict[str, list[dict]] = {}
summaries: dict[str, str] = {}


def get_or_create(run_id: str) -> list[dict]:
    if run_id not in history:
        history[run_id] = []
    return history[run_id]


def add_message(run_id: str, role: str, content: str):
    get_or_create(run_id).append({"role": role, "content": content})


async def maybe_summarize(run_id: str):
    """Summarize older messages if history exceeds threshold."""
    msgs = get_or_create(run_id)
    if len(msgs) < settings.MEMORY_SUMMARIZE_THRESHOLD or not settings.GROQ_API_KEY:
        return

    old = msgs[: -settings.MEMORY_WINDOW]
    existing = summaries.get(run_id, "")
    text_to_summarize = ""
    if existing:
        text_to_summarize += f"Previous summary: {existing}\n\n"
    text_to_summarize += "\n".join(f"{m['role']}: {m['content'][:200]}" for m in old)

    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage

        http_client = httpx.AsyncClient(verify=False)
        llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=settings.GROQ_API_KEY, temperature=0, http_async_client=http_client)
        resp = await llm.ainvoke([HumanMessage(content=f"Summarize this conversation in 2-3 sentences:\n\n{text_to_summarize}")])
        summaries[run_id] = resp.content
        history[run_id] = msgs[-settings.MEMORY_WINDOW :]
        logger.info("Summarized chat history for run %s", run_id)
    except Exception as e:
        logger.warning("Summary failed: %s", e)


def build_memory_prompt(run_id: str) -> str:
    """Build conversation memory section for the LLM prompt."""
    parts = []
    summary = summaries.get(run_id, "")
    if summary:
        parts.append(f"[Summary of earlier conversation]\n{summary}")

    recent = get_or_create(run_id)[-settings.MEMORY_WINDOW :]
    if recent:
        parts.append("[Recent conversation]")
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            parts.append(f"{role}: {msg['content'][:500]}")

    return "\n".join(parts)
