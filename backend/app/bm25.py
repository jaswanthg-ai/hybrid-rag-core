"""BM25 sparse keyword search.

BM25 (Best Matching 25) scores documents by term frequency, inverse document
frequency, and document length normalization. Industry standard for keyword search.
"""

import math
from collections import Counter

from app.embedder import tokenize


class BM25:
    def __init__(self, chunks: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = [tokenize(c) for c in chunks]
        self.n = len(self.corpus)
        self.avgdl = sum(len(d) for d in self.corpus) / max(self.n, 1)
        self.df: dict[str, int] = Counter()
        for doc in self.corpus:
            for w in set(doc):
                self.df[w] += 1

    def score(self, query: str) -> list[float]:
        """Score all chunks against a query. Returns list of BM25 scores."""
        query_terms = tokenize(query)
        scores = [0.0] * self.n
        for q in query_terms:
            if q not in self.df:
                continue
            idf = math.log((self.n - self.df[q] + 0.5) / (self.df[q] + 0.5) + 1.0)
            for i, doc in enumerate(self.corpus):
                tf = doc.count(q)
                dl = len(doc)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-10))
                scores[i] += idf * num / den
        return scores
