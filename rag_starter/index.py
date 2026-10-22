"""Sentence-aware chunking and a small, dependency-free BM25 index.

BM25 is used here because it is deterministic, runs offline and is easy to test.
In a product build the same `search()` interface can be backed by a vector store
or a hybrid (BM25 + embeddings) retriever without changing the answer layer.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .clean import Doc, tokens

STOPWORDS = set(
    "a an and any are as at be by can do does for from get how i in is it my of on or "
    "the there to what when where which who why will with you your".split()
)
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str


def chunk_doc(doc: Doc, max_words: int = 60, overlap_sentences: int = 1) -> list[Chunk]:
    sentences = [s.strip() for s in _SENT.split(doc.text) if s.strip()]
    chunks, current, n = [], [], 0
    for sent in sentences:
        if current and sum(len(s.split()) for s in current) + len(sent.split()) > max_words:
            chunks.append(Chunk(f"{doc.doc_id}#{n}", doc.doc_id, doc.title, " ".join(current)))
            n += 1
            current = current[-overlap_sentences:] if overlap_sentences else []
        current.append(sent)
    if current:
        chunks.append(Chunk(f"{doc.doc_id}#{n}", doc.doc_id, doc.title, " ".join(current)))
    return chunks


def stem(t: str) -> str:
    """Very light plural folding so 'refunds' matches 'refund'."""
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def terms(text: str) -> list[str]:
    return [stem(t) for t in tokens(text) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tf = [Counter(terms(c.title + " " + c.text)) for c in chunks]
        self.len = [sum(tf.values()) for tf in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 0.0
        df = Counter(t for tf in self.tf for t in tf)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        # unseen query terms get the highest weight: they are the most specific words
        self.oov_idf = math.log(1 + (n + 0.5) / 0.5)

    def weight(self, term: str) -> float:
        return self.idf.get(term, self.oov_idf)

    def score(self, query_terms: list[str], i: int) -> float:
        tf, dl, s = self.tf[i], self.len[i], 0.0
        for t in query_terms:
            if t not in tf:
                continue
            f = tf[t]
            s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
        return s

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = terms(query)
        scored = [(self.chunks[i], self.score(q, i)) for i in range(len(self.chunks))]
        scored = [x for x in scored if x[1] > 0]
        scored.sort(key=lambda x: (-x[1], x[0].chunk_id))
        return scored[:k]
