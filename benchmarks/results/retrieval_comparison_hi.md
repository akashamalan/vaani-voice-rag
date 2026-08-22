# Retrieval approaches — Hindi, 1,485,330 passages

Same corpus, same 95,884 answerable queries, same 100-query sample
(`random.seed(0)`). No conditioning, no subsetting. Recall@10 against the
MSMARCO-XI `is_selected` answer key.

| approach | recall@10 | ms/query | index on disk |
|---|---|---|---|
| binary top-200 + int8 rerank | 0.340 | 399.0 | 612 MB |
| binary top-20k + `bitwise_count` | 0.520 | 113.2 | 612 MB |
| exact int8 linear scan | 0.580 | 2180.0 | 544 MB |
| **HNSW M=32, efSearch=64** | **0.590** | **1.00** | 2561 MB |
| HNSW M=32, efSearch=256 | 0.610 | 3.23 | 2561 MB |

**Headline: 2,180 ms → 1.00 ms, a 2,180x speedup, with no recall lost.**

## Why HNSW slightly exceeds "exact"

It isn't beating exact search. `IndexHNSWFlat` stores full float32 vectors,
while the "exact" baseline scans the int8-quantized copy. The 0.010 gap is
the cost of int8 quantization, not a gain from the graph. Against float32
exact search, HNSW at efSearch=64 would tie, not win.

## Latency breakdown, CPU, warm, single query

Measured by `app/retrieval/vector_store.py --runs 200`. This is the serving
path: no batching, one query at a time, CPU only (the VPS has no GPU).

| stage | p50 | p70 | p100 |
|---|---|---|---|
| embed | 39.73 | 44.07 | 305.50 |
| search | 0.97 | 1.15 | 3.00 |
| fetch | 0.18 | 0.20 | 8.07 |
| **TOTAL** | **41.09** | **45.36** | **307.55** |

Embedding is 97% of retrieval latency. Search and LMDB fetch are together
under 1.2 ms at p50 and are not worth optimizing.

## Corpus note

Recall@10 is 0.580–0.590 at 1.49M passages versus 0.854 at 8,291 in the
chunking benchmark. That is the effect of 179x more distractors, not a
regression. Quote 0.59 for the full corpus.

## Build cost

The 1,485,330-vector HNSW build took **586.9 minutes** (9.8 hours) on 12
threads, versus 8.2 minutes for a 500,000-vector build. The superlinear
blowup is most likely memory pressure during `index.add` plus overnight CPU
downclocking. Budget an overnight run per language, not 30 minutes.
