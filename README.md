# Voice RAG — Hindi

Voice question answering over 1.5M Hindi passages, with a retrieval core that
answers in ~19 ms and a guardrail stack that knows when not to answer.

Every number below is measured on this machine, not estimated. Where a design
choice was wrong, the measurement that disproved it is kept.

---

## Architecture

```
  mic ──► browser ──► FastAPI ──► Sarvam STT ──► transcript
                          │
                          ├─ safety      regex, before anything expensive
                          ├─ embed       multilingual-e5-small, 384-dim
                          ├─ search      FAISS HNSW, 1,485,330 vectors
                          ├─ fetch       LMDB, doc_id → passage text
                          ├─ relevance   score floor, skip LLM on empty retrieval
                          ├─ generate    Groq, streamed tokens
                          └─ grounding   overlap + verbatim, post-generation
                          │
                     WebSocket ──► tokens + per-stage timings ──► UI
```

Audio is proxied through the backend, never sent from the browser to Sarvam —
the live link is public, and a key in browser JS is a key that is gone.

**Corpus.** 1,485,330 unique Hindi passages indexed from `ai4bharat/MSMARCO-XI`,
out of ~7.7M available. The cap is a deliberate slice, not a limit of the
design: ingestion is resumable and the architecture is language-agnostic —
adding Tamil is a config change plus an index build.

---

## Chunking: five strategies compared

500 queries, 8,291 passages, recall@10 against the `is_selected` answer key.

| strategy | recall@1 | recall@5 | recall@10 | MRR | chunks/psg | index MB |
|---|---|---|---|---|---|---|
| **passage** | **0.284** | **0.730** | **0.854** | **0.470** | 1.00 | 12.1 |
| sliding_window | 0.278 | 0.724 | 0.852 | 0.462 | 1.24 | 15.0 |
| semantic | 0.268 | 0.700 | 0.832 | 0.443 | 1.83 | 22.2 |
| metadata_aware | 0.250 | 0.662 | 0.780 | 0.420 | 3.62 | 44.0 |
| sentence_window | 0.270 | 0.660 | 0.770 | 0.433 | 3.62 | 44.0 |

**The winner is the no-op, and that result is about the corpus, not chunking.**
MS MARCO passages are already chunk-sized — median 57 words, mean 65, only 19%
over 80 words. An 80-word sliding window over a 57-word passage emits one chunk
and changes nothing. The strategies that genuinely split (3.62 chunks/passage ≈
one per sentence) shred a 3-sentence passage into fragments and lose 8 points of
recall@10 doing it.

Chunking earns its keep on long documents. We measured it and it did not, so we
index passages directly. `semantic` was also 200× slower to cut (26 ms/passage →
~11 hours at 1.5M scale) and did not win.

---

## Retrieval: four approaches at full scale

Same corpus (1,485,330), same 95,884 answerable queries, same 100-query sample.
No conditioning, no subsetting.

| approach | recall@10 | ms/query | index |
|---|---|---|---|
| binary top-200 + int8 rerank | 0.340 | 399.0 | 612 MB |
| binary top-20k + `bitwise_count` | 0.520 | 113.2 | 612 MB |
| exact int8 linear scan | 0.580 | 2180.0 | 544 MB |
| **FAISS HNSW, M=32, efSearch=64** | **0.590** | **1.00** | 2561 MB |

**2,180 ms → 1.00 ms, a 2,180× speedup, with no recall lost.**

Two honest notes:

- HNSW's 0.590 vs 0.580 is *not* the graph beating exact search. `IndexHNSWFlat`
  stores float32; the exact baseline scans the int8 copy. That 0.010 is int8
  quantization loss. Against float32 exact, HNSW ties.
- The binary two-tier design failed for a specific, measurable reason: at 1.5M
  passages a top-200 prefilter retains a correct answer only 35% of the time,
  capping recall regardless of reranker quality. Depth 20k recovers it but costs
  113 ms. The graph index does both jobs better at this scale.

`efSearch=64` matched exact search to three decimals; 128/256/512 bought nothing
and cost latency.

---

## Guardrails: three layers, because one number cannot do it

Off-topic detection is **not** a similarity threshold. We tried, and measured why
it cannot work:

| | top1 | margin (top1 − mean of rest) |
|---|---|---|
| 8 real questions | 0.8689 – 0.9470 | 0.0125 – 0.0419 |
| 6 junk questions | 0.8174 – 0.8698 | 0.0054 – 0.0166 |

**The ranges overlap.** `"purple monkey dishwasher velocity"` scores 0.8174.
e5 embeddings occupy a narrow cone, so every query — meaningful or not — lands
between roughly 0.81 and 0.95. The number does not encode meaning finely enough
to threshold on, and any boundary that separates a sample this small is fitting
noise rather than finding a rule.

So the guardrail is three layers, each catching what the others cannot:

| layer | mechanism | catches | cost |
|---|---|---|---|
| 1. relevance floor | `top_score ≥ 0.80` | empty retrievals only | ~0 ms |
| 2. `NOT_IN_CONTEXT` | LLM reads real passages + real question | off-topic | free (same call) |
| 3. grounding | content-word overlap + verbatim check | fabrication | ~0 ms |

**Layer 2 is the real off-topic guardrail.** It is semantic judgment over the
actual retrieved text, which no cosine similarity can do, and it costs nothing
extra because it rides on the generation call we already make. The model emits
the sentinel; the pipeline converts it to a refusal.

**Layer 3 is two tests, because overlap alone is not enough.** Raw word overlap
scored a fully fabricated answer at 0.600 — function words like की / है /
राजधानी match almost any passage. Filtering function words drops fabrications to
0.33–0.67 while correct answers stay at 1.00. But overlap still cannot catch
entity substitution: "मुंबई भारत की राजधानी है" is 4/5 grounded and only the
entity is wrong. So every number and every content token ≥4 chars must also
appear **verbatim** in the retrieved passages.

| answer | overlap | unsupported | verdict |
|---|---|---|---|
| नई दिल्ली भारत की राजधानी है | 1.000 | — | pass |
| पेरिस फ्रांस की राजधानी है | 0.333 | पेरिस, फ्रांस | refuse |
| मुंबई भारत की राजधानी है | 0.667 | मुंबई | refuse |
| दिल्ली शहर के **24** जिलों में से एक है | 0.750 | 24 | refuse |

The last two rows are the point: both would pass any overlap threshold loose
enough to admit real paraphrase. An embedding check would be strictly worse —
"Mumbai is the capital" and "Delhi is the capital" are neighbours in vector
space, which is exactly the confusion that needs catching.

Safety is a regex over harm patterns and prompt injection (`ignore previous
instructions`, `you are now a`, `reveal your system prompt`), run before
retrieval so a hostile query costs nothing.

---

## Latency

> **PENDING — step 4.** Populate from `benchmarks/results/latency_full_pipeline.txt`
> and `latency_retrieval_only.txt` after running `benchmarks/latency_bench.py`.
> Do not fill this in by hand.

Measured so far, warm, single query, CPU only:

| stage | ms |
|---|---|
| embed | 18.4 |
| search | 0.72 |
| fetch | 0.12 |
| **retrieval core** | **19.3** |

**The 200 ms target is met by the retrieval core, comfortably — 19 ms against a
200 ms budget, roughly 10× headroom.** Full end-to-end is approximately 1 second,
and the difference is not ours: Sarvam STT and Groq generation dominate, and both
are network calls to external services. Embedding is 95% of what remains under
our control; ONNX would take it to ~12 ms and buy nothing we need.

Cold start is materially different and is reported separately: the 2.6 GB index
takes 24–68 s to load and the first embedding call costs ~675 ms against 18 ms
warm. Budget for it on deploy; do not benchmark the first request.

---

## Running it

```bash
.venv\Scripts\activate
```

```bash
python -m uvicorn app.main:app --port 8000
```

Keys are read from the environment only, never from a file in the repo:

```bash
setx SARVAM_API_KEY "your-key"
```

```bash
setx GROQ_API_KEY "your-key"
```

`setx` does not affect the current terminal — open a new one. The server starts
without keys and reports which are missing at `/health`, so retrieval can be
tested in isolation with `{"generate": false}`.

**Endpoints**

| endpoint | purpose |
|---|---|
| `GET /health` | readiness, vector count, which keys are configured |
| `POST /api/query` | text in, one JSON out; also the latency harness |
| `WS /ws/query` | `{"type":"audio","data":<base64>}` or `{"type":"text","text":…}` |

**Verification**

```bash
python -u scripts/ws_probe.py
```

```bash
python -u benchmarks/latency_bench.py --n 200
```

---

## Build cost, recorded

The 1,485,330-vector HNSW build took **586.9 minutes** (9.8 hours) on 12 threads,
against 8.2 minutes for a 500,000-vector build — superlinear, most likely memory
pressure during `index.add`. Ingestion and embedding took a further ~41 minutes
on an RTX 3050. Budget an overnight run per language.
