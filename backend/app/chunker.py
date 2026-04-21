"""Recursive character text splitter.

Splits on natural boundaries: paragraphs > newlines > sentences > words.
Falls back to hard character splits if no separator works.
"""


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into chunks respecting natural boundaries."""
    if len(text) <= size:
        return [text] if text.strip() else []

    for sep in ["\n\n", "\n", ". ", " "]:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks, current = [], ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = part
        if current.strip():
            chunks.append(current.strip())
        return chunks

    # Hard fallback
    return [
        text[i : i + size].strip()
        for i in range(0, len(text), size - overlap)
        if text[i : i + size].strip()
    ]
