# वाणी — Voice RAG
-----------------------------------------------------------------------------
The Live Link Is : https://meals-advised-your-robbie.trycloudflare.com/     |
                                                                            |   
-----------------------------------------------------------------------------

Voice-enabled Retrieval-Augmented Generation over MSMARCO-XI (Hindi).
Speak a question, get an answer grounded in retrieved passages.

Built for HH Goa 2026 Shortlisting Task 2.

## Pipeline

Voice → Sarvam STT → embed → HNSW search → guardrails → Groq → answer

## Numbers

**Retrieval core: 19.3 ms p50, 55.8 ms p100** — target was 200 ms.

200 queries drawn from the answer key, run one at a time through
`/api/query`, warm, CPU only. Percentiles, not a best-case single run.

| Stage | p50 | p70 | p100 |
|---|---|---|---|
| Safety check | 0.02 | 0.02 | 0.04 |
| Embed question | 18.07 | 19.34 | 54.73 |
| Vector search | 1.06 | 1.25 | 2.13 |
| Fetch passages | 0.12 | 0.13 | 4.43 |
| Relevance gate | 0.00 | 0.00 | 0.01 |
| **Retrieval core** | **19.26** | **20.78** | **55.84** |

Even p100 lands 3.6× inside the budget. Embedding is 94% of it; vector
search over 1.49M passages costs about 1 ms.

Reproduce with `python benchmarks/latency_bench.py --n 200 --no-generate`.

**The index must be resident in RAM.** Measured on the same machine with
only 1.7 GB free, retrieval core rose to 262 ms p50 — vector search alone
went from 1.06 ms to 219 ms, because the 2.6 GB HNSW graph was paging to
disk and every search became random disk reads. That is a deployment
requirement, not a property of the algorithm: give the process enough RAM
to hold the index and the numbers above hold.

Full voice-to-answer is ~1.3 s. Speech-to-text (~700 ms) and
generation (~700 ms) are external API calls and dominate. No system
using hosted STT and a hosted LLM completes end to end in 200 ms;
we report both numbers rather than one.

## Retrieval — what we measured

Same corpus, same 95,884 queries.

| Approach | recall@10 | ms/query |
|---|---|---|
| Binary quantization, top-200 + rerank | 0.340 | 399.0 |
| Binary, top-20k + bitwise_count | 0.520 | 113.2 |
| Exact int8 scan | 0.580 | 2180.0 |
| **FAISS HNSW, efSearch=64** | **0.590** | **1.0** |

We built binary quantization first for RAM efficiency, measured it,
found it cost 24 recall points and 100× the latency of a graph index
at this scale, and switched. HNSW ties exact search at 1/2000th the cost.

## Chunking — five strategies compared

| Strategy | recall@10 |
|---|---|
| Whole passage (baseline) | 0.854 |
| Sliding window, 80/20 overlap | 0.852 |
| Semantic (embedding-distance splits) | 0.832 |
| Metadata-aware | 0.780 |
| Sentence window | 0.770 |

The baseline wins because MSMARCO-XI passages have a median length of
57 words — the corpus is already chunked. Extra splitting adds index
size without adding recall. Semantic chunking cost 18.5 ms/passage,
about 7.7 hours for 1.5M, and did not earn it.

Reporting a negative result honestly beats asserting a positive one.

Indic sentence splitting handles the danda (।) — an English splitter
returns Hindi passages as one blob and silently flattens the comparison.

## Guardrails — three layers

Dense similarity scores cannot separate on-topic from off-topic here.
e5 embeddings occupy a narrow cone; junk scored 0.817 against real
questions at 0.869. No single threshold works. So:

1. **Relevance floor (0.80)** — cheap gate, skips the LLM on empty retrievals
2. **`NOT_IN_CONTEXT` sentinel** — the model reads the actual passages and
   declines. Semantic judgment, no extra call.
3. **Grounding check** — content-word overlap plus verbatim matching on
   numbers and rare tokens, catching entity substitution
   ("Mumbai is the capital" fails against Delhi passages).

Plus regex safety filtering and prompt-injection detection before retrieval.

## Harness

Pydantic schemas on every boundary. Retries with exponential backoff and
jitter, breaking immediately on non-429 4xx. Per-stage timing on every
request, streamed live to the UI. Every path terminates in refusal,
answer, or error — no silent exits.

## Data

`ai4bharat/MSMARCO-XI`, `train/hintrain.parquet`.
1,485,330 unique passages indexed of ~7.7M available Hindi.
Answer key derived from the dataset's own `is_selected` flags.
Architecture is language-agnostic — all 13 languages with train files
are a config change.

## Stack

FastAPI · WebSocket · FAISS HNSW · LMDB · multilingual-e5-small ·
Sarvam Saarika v2.5 · Groq · React + Vite

## Run

    pip install -r requirements.txt
    $env:SARVAM_API_KEY = "..."
    $env:GROQ_API_KEY = "..."
    uvicorn app.main:app --port 8000
    npm run dev --prefix frontend

## Team

Metazord — 4 members

#RAGInGoa
