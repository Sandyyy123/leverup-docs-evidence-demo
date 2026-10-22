from pathlib import Path
from rag_starter.clean import Doc,clean_corpus
from rag_starter.index import BM25Index,chunk_doc
ROOT=Path(__file__).parent
import sys,json
from leverup import ask
def build(corpus_path: Path = ROOT / "data" / "corpus.json"):
    raw = [Doc(**d) for d in json.loads(corpus_path.read_text(encoding="utf-8"))]
    docs, report = clean_corpus(raw)
    chunks = [c for d in docs for c in chunk_doc(d)]
    return BM25Index(chunks), report, len(chunks)


if __name__=='__main__':
    print(json.dumps(ask(' '.join(sys.argv[1:]) or 'What is the maximum leverage?'),indent=2))
