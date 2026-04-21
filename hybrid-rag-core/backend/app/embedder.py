"""TF-IDF dense embeddings with fixed dimensions.

Converts text to fixed-size vectors using Term Frequency × Inverse Document Frequency.
No external model needed — pure Python math.
"""

import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class TFIDFEmbedder:
    """Builds TF-IDF vectors from a corpus. Fixed dimension output for vector DB compatibility."""

    def __init__(self, chunks: list[str], dims: int = 1024):
        self.dims = dims
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}

        doc_freq: Counter = Counter()
        for chunk in chunks:
            for w in set(tokenize(chunk)):
                doc_freq[w] += 1

        n = len(chunks)
        idx = 0
        for word, df in doc_freq.items():
            if df < n and idx < dims:
                self.vocab[word] = idx
                self.idf[word] = math.log((n + 1) / (df + 1)) + 1
                idx += 1

    def embed(self, text: str) -> list[float]:
        """Convert text to a normalized TF-IDF vector."""
        words = tokenize(text)
        tf = Counter(words)
        vec = [0.0] * self.dims
        for word, count in tf.items():
            if word in self.vocab:
                vec[self.vocab[word]] = (count / max(len(words), 1)) * self.idf[word]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_all(self, chunks: list[str]) -> list[list[float]]:
        return [self.embed(c) for c in chunks]
