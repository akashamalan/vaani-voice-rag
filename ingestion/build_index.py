"""
Turn the shards into the two-tier search index.

Outputs:
  binary.npy   uint8 [N, 48]   1 bit per dim. 48 bytes each. LIVES IN RAM.
  int8.npy     int8  [N, 384]  384 bytes each. read from disk, not loaded.
  ids.json     row number -> doc id (the LMDB key)
  index_meta.json

Why two copies of the same vectors:
  Round 1 searches the tiny binary copy. Comparing bits is very fast, so we
  can scan all 1.5M in a few milliseconds. It is rough, so we take 200
  candidates instead of 10.
  Round 2 scores only those 200 against the int8 copy, which is accurate.
  Rough on everything, then careful on a few. That is how you get 1.5M
  passages searched in under 30ms.

Also fixes the resume duplicates. --resume starts with an empty `seen` set,
so passages indexed before a crash get indexed again after it. We drop them
here by doc id — first copy wins.

Usage:
    python -u ingestion/build_index.py --lang hi
"""
import argparse, json
from pathlib import Path

import numpy as np

ART = Path(r"C:\Users\akash\voice-rag-artifacts")


def main(lang, artifacts):
    art = Path(artifacts)
    shards = sorted(art.glob(f"emb_{lang}_*.npy"))
    if not shards:
        raise SystemExit(f"no shards in {art}")

    # ---- pass 1: read ids only, work out what to keep -----------------------
    all_ids, per_shard = [], []
    for p in shards:
        idf = art / p.name.replace("emb_", "ids_").replace(".npy", ".json")
        if not idf.exists():
            raise SystemExit(f"{p.name} has no ids file — incomplete shard")
        ids = json.load(open(idf))
        per_shard.append(ids)
        all_ids.extend(ids)

    seen = set()
    keep_masks, kept_ids = [], []
    for ids in per_shard:
        m = np.zeros(len(ids), dtype=bool)
        for i, d in enumerate(ids):
            if d not in seen:
                seen.add(d)
                m[i] = True
                kept_ids.append(d)
        keep_masks.append(m)

    total, unique = len(all_ids), len(kept_ids)
    dropped = total - unique
    print(f"[dedup] {total:,} vectors -> {unique:,} unique "
          f"({dropped:,} dropped, {dropped/total:.3%})")

    # ---- pass 2: quantize straight into memmaps ----------------------------
    binary = np.lib.format.open_memmap(
        art / "binary.npy", mode="w+", dtype=np.uint8, shape=(unique, 48))
    int8 = np.lib.format.open_memmap(
        art / "int8.npy", mode="w+", dtype=np.int8, shape=(unique, 384))

    off = 0
    for p, m in zip(shards, keep_masks):
        v = np.load(p).astype(np.float32)      # float16 on disk -> float32 to work
        v = v[m]
        n = len(v)
        if n == 0:
            continue

        # sanity: these must be unit length or both quantizers are wrong
        norms = np.linalg.norm(v, axis=1)
        if not (0.99 < norms.mean() < 1.01):
            raise SystemExit(f"{p.name}: vectors not normalized (mean {norms.mean():.4f})")

        # binary: keep the sign of each dim, pack 8 bits per byte
        binary[off:off + n] = np.packbits(v > 0, axis=1)

        # int8: unit vectors sit in [-1, 1], so one flat scale of 127 is right
        int8[off:off + n] = np.clip(np.rint(v * 127), -127, 127).astype(np.int8)

        off += n
        print(f"[pack] {p.name}  {off:,}/{unique:,}")

    assert off == unique, f"packed {off} but expected {unique}"

    binary.flush(); int8.flush()
    json.dump(kept_ids, open(art / "ids.json", "w"))

    mb = lambda b: round(b / (1 << 20), 1)
    meta = {
        "lang": lang,
        "n_vectors": unique,
        "n_dropped_duplicates": dropped,
        "dim": 384,
        "binary_mb": mb(unique * 48),      # in RAM
        "int8_mb": mb(unique * 384),       # on disk, mmap
        "shards": [p.name for p in shards],
    }
    json.dump(meta, open(art / "index_meta.json", "w"), indent=2)

    print(f"\n[done] {unique:,} vectors")
    print(f"       binary {meta['binary_mb']}MB  (held in RAM)")
    print(f"       int8   {meta['int8_mb']}MB  (mmap from disk)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--artifacts", default=str(ART))
    main(**vars(ap.parse_args()))
