"""PDF text extraction using pymupdf (fitz)."""

import fitz


def parse_pdf(content: bytes) -> str:
    """Extract text from PDF bytes. Returns full text with pages joined by newlines."""
    doc = fitz.open(stream=content, filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)
