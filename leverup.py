"""Conservative documentation evidence demonstrator; no LLM and no trading."""
import json,re,hashlib
from pathlib import Path
from rag_starter.clean import Doc
from rag_starter.index import BM25Index,chunk_doc
from rag_starter.answer import answer,ABSTAIN
ROOT=Path(__file__).parent

def ask(question):
    records=json.loads((ROOT/'data/leverup_facts.json').read_text())
    # Facts are intentionally a small authored snapshot, not a full documentation crawl.
    # Fail closed outside the two implemented intents, including compound questions.
    q=question.lower().strip().rstrip('?')
    leverage=re.fullmatch(r'(?:what is (?:the )?(?:maximum|max) leverage(?: on leverup)?|how much leverage does leverup support)',q)
    fees=re.fullmatch(r'(?:what percentage of (?:protocol )?fees (?:go|goes|is returned) to traders|are protocol fees returned to traders)',q)
    if not (leverage or fees):return {'answer':ABSTAIN,'abstained':True,'sources':[]}
    row=records[0 if leverage else 1]
    index=BM25Index([c for r in records for c in chunk_doc(Doc(r['id'],r['title'],r['text']))])
    result=answer(index,'maximum leverage' if leverage else 'protocol fees traders',min_score=0,min_coverage=0)
    # Return the exact chosen evidence record, never a generative synthesis.
    return {'answer':row['text'],'abstained':False,'sources':[{'section':row['title'],'url':row['url'],'snapshot_sha256':hashlib.sha256(row['text'].encode()).hexdigest()}]}
