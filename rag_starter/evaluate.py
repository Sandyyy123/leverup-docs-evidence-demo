"""Golden-set evaluation: retrieval quality and abstention behaviour.

Each golden row: {"question", "expected_doc"} where expected_doc is null for
questions the corpus cannot answer (those must abstain).
Metrics:
* hit_rate@k  - expected document appears in the top-k chunks
* mrr         - mean reciprocal rank of the first chunk from the expected document
* answer_doc_accuracy - cited document equals the expected document
* abstention_precision / recall - on the unanswerable rows
"""
from __future__ import annotations

from .answer import answer
from .index import BM25Index


def evaluate(index: BM25Index, golden: list[dict], k: int = 3) -> dict:
    rows, hits, rr, correct_doc = [], 0, 0.0, 0
    answerable = [g for g in golden if g.get("expected_doc")]
    unanswerable = [g for g in golden if not g.get("expected_doc")]
    abstained_on_unanswerable = abstained_on_answerable = 0

    for g in golden:
        ranked = index.search(g["question"], k=k)
        docs = [c.doc_id for c, _ in ranked]
        a = answer(index, g["question"], k=k)
        cited_doc = a.citations[0].split("#")[0] if a.citations else None
        exp = g.get("expected_doc")
        if exp:
            if exp in docs:
                hits += 1
                rr += 1 / (docs.index(exp) + 1)
            correct_doc += int(cited_doc == exp)
            abstained_on_answerable += int(a.abstained)
        else:
            abstained_on_unanswerable += int(a.abstained)
        rows.append({
            "question": g["question"], "expected_doc": exp, "retrieved": docs,
            "answer": a.text, "abstained": a.abstained,
            "top_score": round(a.top_score, 3), "coverage": round(a.coverage, 3),
        })

    total_abstained = abstained_on_unanswerable + abstained_on_answerable
    n_ans = len(answerable) or 1
    return {
        "n_questions": len(golden),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        f"hit_rate@{k}": round(hits / n_ans, 3),
        "mrr": round(rr / n_ans, 3),
        "answer_doc_accuracy": round(correct_doc / n_ans, 3),
        "abstention_recall": round(abstained_on_unanswerable / (len(unanswerable) or 1), 3),
        "abstention_precision": round(abstained_on_unanswerable / total_abstained, 3) if total_abstained else None,
        "rows": rows,
    }
