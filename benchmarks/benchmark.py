
import argparse, hashlib, json, sys, time
from pathlib import Path

# must come before any `ingestion.*` import — running this file puts
# benchmarks/ on sys.path, not the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ingestion.loader import iter_rows, extract_passages
from ingestion.chunking import get_chunker, STRATEGIES

MODEL = "intfloat/multilingual-e5-small"
OUT = Path("benchmarks/results"); OUT.mkdir(parents=True, exist_ok=True)


def pid(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


# ------------------------------------------------------------------ load slice
def load_slice(lang: str, n_queries: int):
    """
    corpus  {passage_id: text}       right answers plus distractors
    qrels   [(query, {right_ids})]   the answer key
    """
    corpus, qrels = {}, []

    for row in iter_rows(lang):
        passages = extract_passages(row)
        if not passages:
            continue

        right = set()
        for text, is_sel in passages:
            i = pid(text)
            corpus[i] = text
            if is_sel:
                right.add(i)

        q = (row.get("query") or "").strip()
        if q and right:
            qrels.append((q, right))

        if len(qrels) >= n_queries:
            break

    if not corpus:
        raise SystemExit("empty corpus — extraction is broken, do not trust results")

    print(f"[slice] {len(qrels)} queries, {len(corpus):,} passages")
    return corpus, qrels


# -------------------------------------------------------------------- one run
def run_strategy(name, corpus, qrels, model, k_list=(1, 5, 10, 20)):
    chunker = get_chunker(name, encoder=model)

    # 1. cut everything
    t0 = time.perf_counter()
    texts, owners = [], []          # owners[i] = which passage chunk i belongs to
    for i, passage in corpus.items():
        for c in chunker.chunk(passage, {"lang": "xx", "pid": i}):
            texts.append(c.text)
            owners.append(i)
    cut_s = time.perf_counter() - t0

    # 2. numbers
    t0 = time.perf_counter()
    with torch.inference_mode():
        vecs = model.encode([f"passage: {t}" for t in texts],
                            batch_size=256, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)
    vecs = vecs.astype(np.float32)
    embed_s = time.perf_counter() - t0

    # 3. ask the questions
    queries = [q for q, _ in qrels]
    with torch.inference_mode():
        qv = model.encode([f"query: {q}" for q in queries],
                          batch_size=256, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False)
    qv = qv.astype(np.float32)

    owners = np.array(owners)
    kmax = max(k_list)
    hits = {k: 0 for k in k_list}
    mrr = 0.0
    search_ms = []

    for qi in range(len(queries)):
        t0 = time.perf_counter()
        scores = vecs @ qv[qi]                    # every chunk, one number each
        top = np.argpartition(-scores, kmax * 4)[:kmax * 4]
        top = top[np.argsort(-scores[top])]

        # many chunks can point at the same passage — keep first, drop repeats
        seen, ranked = set(), []
        for idx in top:
            o = owners[idx]
            if o not in seen:
                seen.add(o)
                ranked.append(o)
            if len(ranked) >= kmax:
                break
        search_ms.append((time.perf_counter() - t0) * 1000)

        right = qrels[qi][1]
        for k in k_list:
            if right & set(ranked[:k]):
                hits[k] += 1
        for rank, o in enumerate(ranked, 1):
            if o in right:
                mrr += 1.0 / rank
                break

    n = len(queries)
    return {
        "strategy": name,
        "chunks": len(texts),
        "chunks_per_passage": round(len(texts) / len(corpus), 2),
        "index_mb": round(len(texts) * 384 * 4 / (1 << 20), 1),
        **{f"recall@{k}": round(hits[k] / n, 4) for k in k_list},
        "mrr": round(mrr / n, 4),
        "cut_s": round(cut_s, 1),
        "embed_s": round(embed_s, 1),
        "search_ms_p50": round(float(np.percentile(search_ms, 50)), 2),
    }


# ------------------------------------------------------------------------ main
def main(lang, queries, strategies):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL, device=dev)
    if dev == "cuda":
        model = model.half()
    model.max_seq_length = 192
    print(f"[init] device={dev}")

    corpus, qrels = load_slice(lang, queries)

    rows = []
    for s in strategies:
        print(f"\n[run] {s}")
        r = run_strategy(s, corpus, qrels, model)
        rows.append(r)
        print("     ", {k: v for k, v in r.items() if k != "strategy"})

    rows.sort(key=lambda r: -r["recall@10"])

    cols = ["strategy", "recall@1", "recall@5", "recall@10", "mrr",
            "chunks_per_passage", "index_mb", "search_ms_p50"]
    print("\n" + " | ".join(f"{c:>18s}" for c in cols))
    print("-" * (21 * len(cols)))
    for r in rows:
        print(" | ".join(f"{str(r[c]):>18s}" for c in cols))

    best = rows[0]["strategy"]
    print(f"\nWINNER: {best}   <- this is what the live site should use")

    out = OUT / f"chunking_{lang}.json"
    json.dump({"lang": lang, "n_queries": len(qrels),
               "n_passages": len(corpus), "results": rows},
              open(out, "w"), indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    main(**vars(ap.parse_args()))