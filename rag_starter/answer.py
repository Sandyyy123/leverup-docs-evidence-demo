"""Grounded answering with citations and abstention.

Rules enforced in code (and tested):
* Every answer sentence comes from a retrieved chunk and carries its citation.
* If the best retrieval score is below `min_score`, or the IDF-weighted share of
  query terms found in the evidence is below `min_coverage`, the system abstains
  instead of guessing. Words never seen in the corpus (e.g. a product name that
  is not documented) carry the highest weight, so they pull coverage down.
* An optional `generator` (any LLM call) may rephrase the evidence, but its
  output is accepted only when its text is an exact extract of retrieved evidence and its citations are allowed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .index import BM25Index, terms

ABSTAIN = "I could not find this in the provided documents."
_CITE = re.compile(r"\[([^\]]+#\d+)\]")


@dataclass
class Answer:
    question: str
    text: str
    citations: list[str] = field(default_factory=list)
    abstained: bool = False
    top_score: float = 0.0
    coverage: float = 0.0


def answer(
    index: BM25Index,
    question: str,
    k: int = 3,
    min_score: float = 2.0,
    min_coverage: float = 0.4,
    max_sentences: int = 2,
    generator: Callable[[str, list[tuple[str, str]]], str] | None = None,
) -> Answer:
    hits = index.search(question, k=k)
    q_terms = set(terms(question))
    if not hits or not q_terms:
        return Answer(question, ABSTAIN, abstained=True)
    top = hits[0][1]
    evidence_terms = set()
    for c, _ in hits:
        evidence_terms |= set(terms(c.title + " " + c.text))
    total = sum(index.weight(t) for t in q_terms)
    coverage = sum(index.weight(t) for t in q_terms & evidence_terms) / total
    if top < min_score or coverage < min_coverage:
        return Answer(question, ABSTAIN, abstained=True, top_score=top, coverage=coverage)

    context = [(c.chunk_id, c.text) for c, _ in hits]
    if generator is not None:
        draft = generator(question, context)
        cited = set(_CITE.findall(draft))
        allowed = {cid for cid, _ in context}
        evidence_text = " ".join(text for _, text in context)
        plain = _CITE.sub("", draft).strip()
        if cited and cited <= allowed and plain and plain in evidence_text:
            return Answer(question, draft, sorted(cited), False, top, coverage)
        # fall through to extractive answer when the generator is ungrounded

    best = hits[0][0]
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", best.text) if s]
    score = {i: sum(index.weight(t) for t in q_terms & set(terms(s))) for i, s in enumerate(sentences)}
    picked = sorted(sorted(score, key=lambda i: -score[i])[:max_sentences])  # keep document order
    chosen = " ".join(sentences[i] for i in picked if score[i] > 0)
    return Answer(question, f"{chosen} [{best.chunk_id}]", [best.chunk_id], False, top, coverage)
