"""
Lexical retrieval over the same passages the dense index holds.

Why this exists: dense recall@10 measured 0.530, and top_score was 0.856-0.956
on every query INCLUDING every miss — the score cannot tell a hit from a miss,
so there is no dense-side knob to turn. Lexical retrieval fails differently
from dense retrieval (exact term match vs semantic proximity), which is the
only reason fusing them can beat either alone.

MEASURED OUTCOME: RRF hybrid reached 0.590 recall@10 vs 0.530 dense-at-k10,
but plain dense at k=20 reached 0.600 for no extra latency, while BM25 search
alone costs 28ms p50 / 203ms p100. Dense k=20 dominates. This module is kept
as the evidence for that decision, not as the shipping retrieval path.

No re-embedding and no reindex: the passage text is already in passages.lmdb,
keyed by the same doc_id the dense index returns.

Build (about 10 minutes, memory-hungry):
    python -m app.retrieval.bm25 --build
"""
import json
import re
from pathlib import Path

import bm25s
import lmdb

from app.config import ARTIFACTS, LANG

# Devanagari U+0900-U+097F named explicitly, plus ASCII alphanumerics for the
# English proper nouns and figures scattered through these passages. The
# default \w+ splitter would silently mangle Hindi and produce an index that
# looks fine and matches nothing.
TOKEN_RE = r"[0-9A-Za-zऀ-ॿ]+"
_TOKEN = re.compile(TOKEN_RE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def index_dir(art: Path = None, lang: str = LANG) -> Path:
    return Path(art or ARTIFACTS) / f"bm25_{lang}"


class BM25Index:
    """Thin wrapper: keeps doc_ids aligned with bm25s' internal row order."""

    def __init__(self, retriever, doc_ids: list[str]):
        self.retriever = retriever
        self.doc_ids = doc_ids

    @classmethod
    def load(cls, art: Path = None, lang: str = LANG) -> "BM25Index":
        d = index_dir(art, lang)
        retriever = bm25s.BM25.load(str(d), load_corpus=False)
        doc_ids = json.loads((d / "doc_ids.json").read_text(encoding="utf-8"))
        return cls(retriever, doc_ids)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        toks = tokenize(query)
        if not toks:
            return []
        k = min(k, len(self.doc_ids))
        idx, scores = self.retriever.retrieve([toks], k=k, show_progress=False)
        return [(self.doc_ids[int(i)], float(s))
                for i, s in zip(idx[0], scores[0])]


def build(art: Path = None, lang: str = LANG) -> Path:
    art = Path(art or ARTIFACTS)
    env = lmdb.open(str(art / "passages.lmdb"), readonly=True, lock=False,
                    readahead=False)

    doc_ids, corpus = [], []
    with env.begin() as txn:
        for key, val in txn.cursor():
            doc_ids.append(key.decode())
            corpus.append(json.loads(val)["t"])
    env.close()
    print(f"[bm25] read {len(corpus):,} passages from lmdb")

    # stopwords defaults to "english" — must be emptied, or English stopwords
    # are stripped from the Latin tokens that appear throughout these Hindi
    # passages. No stemmer: PyStemmer has no Hindi, and stemming the Latin
    # half only would make the two scripts inconsistent.
    tokens = bm25s.tokenize(corpus, token_pattern=TOKEN_RE, stopwords=[],
                            stemmer=None, show_progress=True)
    del corpus
    print("[bm25] tokenized")

    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=True)
    print("[bm25] indexed")

    out = index_dir(art, lang)
    retriever.save(str(out), corpus=None)
    (out / "doc_ids.json").write_text(json.dumps(doc_ids), encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"[bm25] saved -> {out}  ({size/1e6:.0f} MB, {len(doc_ids):,} docs)")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    else:
        ix = BM25Index.load()
        for did, sc in ix.search("भारत की राजधानी क्या है", k=5):
            print(f"  {sc:8.3f}  {did}")
