"""
Turn the dataset into numbers on disk.

Outputs into artifacts/:
  emb_{lang}_{shard}.npy   float16 [N, 384], unit length
  ids_{lang}_{shard}.json  doc ids for that shard, same order
  passages.lmdb            doc_id -> {text, lang}
  qrels_{lang}.jsonl       query -> right doc ids   (the answer key)
  state_{lang}.json        resume point

Usage:
    python -u scripts/build_dataset.py --lang hi --max-passages 1500000
    python -u scripts/build_dataset.py --lang hi --resume      # after a crash
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import lmdb
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingestion.loader import iter_rows, extract_passages

# NOT inside the project — the project lives in OneDrive, and a live
# memory-mapped LMDB under a sync client is a corruption path.
ART = Path.home() / "voice-rag-artifacts"; ART.mkdir(exist_ok=True)
MODEL = "intfloat/multilingual-e5-small"
BATCH = 256          # lower to 128 if CUDA runs out of memory
SHARD = 200_000      # passages per .npy file
LMDB_MAP = 12 << 30


def doc_id(text: str) -> str:
    """Same text -> same id. This is how duplicates get dropped."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


def main(lang, max_passages, resume, use_english):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("!! running on CPU. this will take 10+ hours. stop and fix torch.")
    print(f"[init] device={dev}")

    model = SentenceTransformer(MODEL, device=dev)
    if dev == "cuda":
        model = model.half()
    model.max_seq_length = 192

    env = lmdb.open(str(ART / "passages.lmdb"), map_size=LMDB_MAP, writemap=True)

    state_f = ART / f"state_{lang}.json"
    state = json.loads(state_f.read_text()) if (resume and state_f.exists()) else {
        "shard": 0, "rows_consumed": 0, "n_passages": 0}
    if resume:
        print(f"[resume] skipping {state['rows_consumed']:,} rows")

    seen = set()
    buf_txt, buf_ids = [], []
    shard_vecs, shard_ids = [], []
    qrels = open(ART / f"qrels_{lang}.jsonl", "a" if resume else "w", encoding="utf-8")

    t0 = time.time()
    n = state["n_passages"]
    raw_seen = 0

    def flush_embed():
        if not buf_txt:
            return
        with torch.inference_mode():
            v = model.encode([f"passage: {t}" for t in buf_txt],
                             batch_size=BATCH, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)
        shard_vecs.append(v.astype(np.float16))
        shard_ids.extend(buf_ids)
        buf_txt.clear(); buf_ids.clear()

    def flush_shard():
        if not shard_vecs:
            return
        arr = np.concatenate(shard_vecs, axis=0)
        np.save(ART / f"emb_{lang}_{state['shard']:04d}.npy", arr)
        json.dump(shard_ids, open(ART / f"ids_{lang}_{state['shard']:04d}.json", "w"))
        print(f"[shard] {state['shard']:04d}  n={len(arr):,}  total={n:,}  "
              f"{time.time()-t0:.0f}s")
        state["shard"] += 1
        shard_vecs.clear(); shard_ids.clear()
        state_f.write_text(json.dumps(state))

    txn = env.begin(write=True)
    pending = 0

    try:
        for row_i, row in enumerate(iter_rows(lang, skip=state["rows_consumed"]),
                                    start=state["rows_consumed"]):
            state["rows_consumed"] = row_i + 1
            passages = extract_passages(row, use_english=use_english)
            raw_seen += len(passages)
            right = []

            for text, is_sel in passages:
                did = doc_id(text)
                if is_sel:
                    right.append(did)
                if did in seen:
                    continue
                seen.add(did)

                txn.put(did.encode(),
                        json.dumps({"t": text, "l": lang},
                                   ensure_ascii=False).encode())
                pending += 1

                buf_txt.append(text); buf_ids.append(did)
                n += 1
                state["n_passages"] = n

                if len(buf_txt) >= BATCH * 8:
                    flush_embed()
                if sum(len(v) for v in shard_vecs) >= SHARD:
                    flush_shard()

            q = (row.get("query") or "").strip()
            if q and right:
                qrels.write(json.dumps({"q": q, "rel": right},
                                       ensure_ascii=False) + "\n")

            if pending >= 20_000:
                txn.commit(); txn = env.begin(write=True); pending = 0

            if n >= max_passages:
                print(f"[stop] reached max-passages={max_passages:,}")
                break

            if row_i % 20_000 == 0 and row_i:
                el = time.time() - t0
                print(f"[prog] rows={row_i:,} unique={n:,} raw={raw_seen:,} "
                      f"dedup={1 - n/max(1, raw_seen):.0%} "
                      f"{n/max(el,1):.0f}/s  {el/60:.1f}m")

    except KeyboardInterrupt:
        print("\n[stop] saving. rerun with --resume to continue.")
    finally:
        flush_embed()
        flush_shard()
        txn.commit(); env.sync(); env.close(); qrels.close()
        state_f.write_text(json.dumps(state))
        print(f"[done] {lang}: {n:,} unique passages from {raw_seen:,} raw, "
              f"{state['shard']} shards, {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--max-passages", type=int, default=1_500_000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--use-english", action="store_true",
                    help="index English_passages instead of Translated_passages")
    main(**vars(ap.parse_args()))
