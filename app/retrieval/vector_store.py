"""
The search wrapper. Everything above this (FastAPI, guardrails, the LLM)
talks to this class and nothing else.

It does four things and times each one:
    1. turn the question into numbers
    2. search the graph index
    3. pull the passage text out of LMDB
    4. hand back results plus timings

The timings are not for debugging. They are the numbers that go on the
latency panel in the UI and into your P50/P70/P100 table, so they get
measured properly and returned with every single query.

Load once at server startup, never per request:
    store = VectorStore(lang="hi")
    store.load()
"""
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import lmdb
import numpy as np
from sentence_transformers import SentenceTransformer

ART = Path(r"C:\Users\akash\voice-rag-artifacts")
MODEL = "intfloat/multilingual-e5-small"


@dataclass
class Hit:
    doc_id: str
    text: str
    score: float          # cosine similarity, roughly -1 to 1
    rank: int


@dataclass
class SearchResult:
    query: str
    hits: list[Hit]
    timings_ms: dict = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return round(sum(self.timings_ms.values()), 2)

    @property
    def top_score(self) -> float:
        """Used by the guardrails. Low top score means we found nothing good."""
        return self.hits[0].score if self.hits else 0.0


class VectorStore:
    def __init__(self, lang: str = "hi", artifacts: Path = ART,
                 ef_search: int = 64, device: str = "cpu"):
        self.lang = lang
        self.art = Path(artifacts)
        self.ef_search = ef_search
        self.device = device
        self.index = None
        self.ids = None
        self.env = None
        self.model = None

    # ------------------------------------------------------------- startup
    def load(self):
        t0 = time.time()

        self.model = SentenceTransformer(MODEL, device=self.device)
        self.model.max_seq_length = 192
        # one warm-up pass — the first encode is always slow and we do not
        # want that landing on a real user
        self.model.encode(["query: warm up"], normalize_embeddings=True)

        self.index = faiss.read_index(str(self.art / f"hnsw_{self.lang}.faiss"))
        self.index.hnsw.efSearch = self.ef_search

        self.ids = json.load(open(self.art / f"ids_ann_{self.lang}.json"))

        self.env = lmdb.open(str(self.art / "passages.lmdb"),
                             readonly=True, lock=False, readahead=False)

        if self.index.ntotal != len(self.ids):
            raise SystemExit(
                f"index has {self.index.ntotal:,} vectors but ids.json has "
                f"{len(self.ids):,} — these must match or every result is wrong")

        print(f"[store] {self.index.ntotal:,} vectors, efSearch={self.ef_search}, "
              f"loaded in {time.time()-t0:.1f}s")
        return self

    # --------------------------------------------------------------- search
    @contextmanager
    def _timed(self, timings: dict, name: str):
        t0 = time.perf_counter()
        yield
        timings[name] = round((time.perf_counter() - t0) * 1000, 2)

    def search(self, query: str, k: int = 5) -> SearchResult:
        timings = {}

        with self._timed(timings, "embed"):
            # e5 needs "query: " on questions and "passage: " on documents.
            # they were indexed with "passage: ". mismatching these quietly
            # wrecks recall, so it is not optional.
            qv = self.model.encode([f"query: {query}"],
                                   normalize_embeddings=True,
                                   convert_to_numpy=True).astype(np.float32)

        with self._timed(timings, "search"):
            scores, idx = self.index.search(qv, k)

        with self._timed(timings, "fetch"):
            hits = []
            with self.env.begin() as txn:
                for rank, (i, s) in enumerate(zip(idx[0], scores[0])):
                    if i < 0:                      # faiss pads with -1
                        continue
                    doc = self.ids[i]
                    raw = txn.get(doc.encode())
                    if raw is None:                # id in index, text missing
                        continue
                    hits.append(Hit(doc_id=doc,
                                    text=json.loads(raw)["t"],
                                    score=float(s),
                                    rank=rank))

        return SearchResult(query=query, hits=hits, timings_ms=timings)

    def close(self):
        if self.env:
            self.env.close()


# ------------------------------------------------------------------- check
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--q", default="भारत की राजधानी क्या है")
    ap.add_argument("--runs", type=int, default=200)
    a = ap.parse_args()

    store = VectorStore(lang=a.lang).load()

    r = store.search(a.q, k=3)
    print(f"\nquery: {r.query}")
    for h in r.hits:
        print(f"  [{h.rank}] {h.score:.3f}  {h.text[:100]}")
    print(f"timings: {r.timings_ms}  total {r.total_ms}ms")

    # real percentiles, warm, one query at a time — the way it actually serves
    print(f"\nrunning {a.runs} queries for percentiles...")
    per_stage = {}
    totals = []
    for _ in range(a.runs):
        r = store.search(a.q, k=5)
        totals.append(r.total_ms)
        for k, v in r.timings_ms.items():
            per_stage.setdefault(k, []).append(v)

    for stage, vals in per_stage.items():
        print(f"  {stage:8s} p50 {np.percentile(vals,50):6.2f}  "
              f"p70 {np.percentile(vals,70):6.2f}  "
              f"p100 {np.percentile(vals,100):6.2f} ms")
    print(f"  {'TOTAL':8s} p50 {np.percentile(totals,50):6.2f}  "
          f"p70 {np.percentile(totals,70):6.2f}  "
          f"p100 {np.percentile(totals,100):6.2f} ms")
