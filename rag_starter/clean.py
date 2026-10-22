"""Document cleaning before indexing.

Steps (each one is counted in the returned report so nothing is dropped silently):
1. Unicode NFKC normalisation and removal of control / zero-width characters.
2. Whitespace collapse and removal of repeated boilerplate lines
   (lines that appear in at least `boilerplate_min_docs` documents, e.g. footers).
3. Drop documents that are empty or too short to answer anything.
4. Exact duplicate removal (hash of normalised text).
5. Near-duplicate removal (Jaccard similarity of word 3-shingles).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)
_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class Doc:
    doc_id: str
    title: str
    text: str


@dataclass
class CleanReport:
    input_docs: int = 0
    kept_docs: int = 0
    boilerplate_lines_removed: int = 0
    dropped_short: list[str] = field(default_factory=list)
    dropped_exact_duplicate: list[str] = field(default_factory=list)
    dropped_near_duplicate: list[tuple[str, str, float]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "input_docs": self.input_docs,
            "kept_docs": self.kept_docs,
            "boilerplate_lines_removed": self.boilerplate_lines_removed,
            "dropped_short": self.dropped_short,
            "dropped_exact_duplicate": self.dropped_exact_duplicate,
            "dropped_near_duplicate": [
                {"doc_id": d, "duplicate_of": k, "jaccard": round(s, 3)}
                for d, k, s in self.dropped_near_duplicate
            ],
        }


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = tokens(text)
    return {tuple(words[i : i + n]) for i in range(max(len(words) - n + 1, 1))}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def clean_corpus(
    docs: list[Doc],
    min_words: int = 12,
    near_dup_threshold: float = 0.8,
    boilerplate_min_docs: int = 3,
) -> tuple[list[Doc], CleanReport]:
    report = CleanReport(input_docs=len(docs))
    normed = [Doc(d.doc_id, d.title.strip(), normalise(d.text)) for d in docs]

    line_counts = Counter(ln for d in normed for ln in set(d.text.splitlines()))
    boiler = {ln for ln, c in line_counts.items() if c >= boilerplate_min_docs}
    stripped = []
    for d in normed:
        kept = [ln for ln in d.text.splitlines() if ln not in boiler]
        report.boilerplate_lines_removed += len(d.text.splitlines()) - len(kept)
        stripped.append(Doc(d.doc_id, d.title, "\n".join(kept)))

    kept_docs: list[Doc] = []
    seen_hash: dict[str, str] = {}
    kept_shingles: list[tuple[str, set]] = []
    for d in stripped:
        if len(tokens(d.text)) < min_words:
            report.dropped_short.append(d.doc_id)
            continue
        h = hashlib.sha256(" ".join(tokens(d.text)).encode()).hexdigest()
        if h in seen_hash:
            report.dropped_exact_duplicate.append(d.doc_id)
            continue
        sh = shingles(d.text)
        best = max(((jaccard(sh, s), k) for k, s in kept_shingles), default=(0.0, ""))
        if best[0] >= near_dup_threshold:
            report.dropped_near_duplicate.append((d.doc_id, best[1], best[0]))
            continue
        seen_hash[h] = d.doc_id
        kept_shingles.append((d.doc_id, sh))
        kept_docs.append(d)
    report.kept_docs = len(kept_docs)
    return kept_docs, report
