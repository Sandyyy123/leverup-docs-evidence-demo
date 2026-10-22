import json
from pathlib import Path

import pytest

from main import build
from rag_starter.answer import ABSTAIN, answer
from rag_starter.clean import Doc, clean_corpus, normalise
from rag_starter.evaluate import evaluate
from rag_starter.index import BM25Index, chunk_doc, stem

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built():
    return build()


def golden():
    return [json.loads(l) for l in (ROOT / "data" / "golden.jsonl").read_text().splitlines() if l.strip()]


def test_normalise_removes_zero_width_and_extra_spaces():
    assert normalise("Coming​   soon.\n\n  x ") == "Coming soon.\nx"


def test_cleaning_report_counts_every_drop(built):
    _, report, _ = built
    r = report.as_dict()
    assert r["input_docs"] == 9 and r["kept_docs"] == 6
    assert r["dropped_short"] == ["stub"]
    assert r["dropped_exact_duplicate"] == ["api_limits_copy"]
    assert [d["doc_id"] for d in r["dropped_near_duplicate"]] == ["export_v1"]
    kept = r["kept_docs"] + len(r["dropped_short"]) + len(r["dropped_exact_duplicate"]) + len(r["dropped_near_duplicate"])
    assert kept == r["input_docs"]


def test_boilerplate_footer_removed(built):
    index, _, _ = built
    assert all("Contact support" not in c.text for c in index.chunks)


def test_short_doc_kept_when_threshold_lowered():
    docs, rep = clean_corpus([Doc("a", "A", "one two three")], min_words=2)
    assert len(docs) == 1 and rep.dropped_short == []


def test_chunk_ids_are_stable_and_overlap():
    d = Doc("x", "X", " ".join(f"Sentence {i} has five words." for i in range(30)))
    chunks = chunk_doc(d, max_words=20, overlap_sentences=1)
    assert [c.chunk_id for c in chunks][:2] == ["x#0", "x#1"]
    assert chunks[0].text.split(". ")[-1].rstrip(".") in chunks[1].text


def test_stem_folds_plurals_only():
    assert stem("refunds") == "refund" and stem("policies") == "policy" and stem("access") == "access"


def test_answer_cites_retrieved_chunk(built):
    index, _, _ = built
    a = answer(index, "How long do data export download links stay valid?")
    assert not a.abstained
    assert "7 days" in a.text and a.citations == ["export#0"]
    assert a.text.endswith("[export#0]")


@pytest.mark.parametrize("q", [
    "Does Acme Workspace integrate with Salesforce?",
    "What is the uptime SLA guarantee?",
    "Is there a student discount?",
])
def test_unanswerable_questions_abstain(built, q):
    index, _, _ = built
    a = answer(index, q)
    assert a.abstained and a.text == ABSTAIN and a.citations == []


def test_generator_with_unknown_citation_is_rejected(built):
    index, _, _ = built
    fake_llm = lambda q, ctx: "Links last forever. [made_up#9]"
    a = answer(index, "How long do data export download links stay valid?", generator=fake_llm)
    assert "forever" not in a.text and a.citations == ["export#0"]


def test_generator_with_valid_citation_is_accepted(built):
    index, _, _ = built
    good_llm = lambda q, ctx: f"Download links expire after 7 days for security reasons. [{ctx[0][0]}]"
    a = answer(index, "How long do data export download links stay valid?", generator=good_llm)
    assert a.text.startswith("Download links expire") and a.citations == ["export#0"]


def test_empty_index_abstains():
    a = answer(BM25Index([]), "anything at all?")
    assert a.abstained


def test_golden_set_metrics(built):
    index, _, _ = built
    m = evaluate(index, golden(), k=3)
    assert m["n_questions"] == 13 and m["n_answerable"] == 10
    assert m["hit_rate@3"] == 1.0
    assert m["answer_doc_accuracy"] == 1.0
    assert m["abstention_recall"] == 1.0 and m["abstention_precision"] == 1.0
