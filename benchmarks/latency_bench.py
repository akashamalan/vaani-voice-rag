"""
Submission latency numbers.

Loops real questions from qrels_hi.jsonl through /api/query and reports
P50/P70/P100 per stage, plus two totals that mean different things:

    retrieval core   embed + search + fetch   — what we control
    full pipeline    total_ms                 — includes Groq generation

Cold and warm are reported separately. The first requests after startup hit
an unwarmed embedding model and page-fault the 2.6GB index off disk; mixing
them into one percentile hides both numbers.

Usage (server must already be running):
    python -u benchmarks/latency_bench.py --n 200
    python -u benchmarks/latency_bench.py --n 200 --no-generate   # retrieval only
"""
import argparse
import json
import random
import time
from pathlib import Path

import httpx
import numpy as np

ART = Path.home() / "voice-rag-artifacts"
OUT = Path("benchmarks/results")
STAGES = ["safety", "embed", "search", "fetch", "relevance", "generate", "grounding"]
CORE = ["embed", "search", "fetch"]


def pct(v, p):
    return float(np.percentile(v, p)) if v else float("nan")


def table(title, rows: dict, totals: dict) -> str:
    out = [f"\n{title}", f"{'stage':<16}{'p50':>9}{'p70':>9}{'p100':>9}{'n':>7}"]
    for s in STAGES:
        v = rows.get(s) or []
        if not v:
            continue
        out.append(f"{s:<16}{pct(v,50):>9.2f}{pct(v,70):>9.2f}{pct(v,100):>9.2f}{len(v):>7}")
    out.append("-" * 50)
    for name, v in totals.items():
        if v:
            out.append(f"{name:<16}{pct(v,50):>9.2f}{pct(v,70):>9.2f}{pct(v,100):>9.2f}{len(v):>7}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/api/query")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--cold", type=int, default=5, help="first N counted as cold")
    ap.add_argument("--no-generate", action="store_true")
    a = ap.parse_args()

    qs = []
    with open(ART / "qrels_hi.jsonl", encoding="utf-8") as f:
        for line in f:
            qs.append(json.loads(line)["q"])
            if len(qs) > 20000:
                break
    random.seed(11)
    sample = random.sample(qs, a.n)
    print(f"{a.n} queries from qrels_hi.jsonl | generate={not a.no_generate}")

    cold = {s: [] for s in STAGES}
    warm = {s: [] for s in STAGES}
    cold_core, warm_core, cold_tot, warm_tot = [], [], [], []
    refused = {}
    fails = 0

    with httpx.Client(timeout=120) as c:
        for i, q in enumerate(sample):
            try:
                r = c.post(a.url, json={"text": q, "generate": not a.no_generate})
                d = r.json()
            except Exception as exc:                            # noqa: BLE001
                fails += 1
                print(f"  [{i}] request failed: {type(exc).__name__}")
                continue

            if d.get("refused"):
                refused[d.get("refusal_reason")] = refused.get(d.get("refusal_reason"), 0) + 1

            t = d.get("timings_ms") or {}
            bucket, core_b, tot_b = ((cold, cold_core, cold_tot) if i < a.cold
                                     else (warm, warm_core, warm_tot))
            for s in STAGES:
                if s in t:
                    bucket[s].append(t[s])
            core = sum(t.get(s, 0.0) for s in CORE)
            if core:
                core_b.append(core)
            if d.get("total_ms"):
                tot_b.append(d["total_ms"])

            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{a.n}  warm core p50 "
                      f"{pct(warm_core,50):.2f}ms  full p50 {pct(warm_tot,50):.2f}ms")

    report = [f"latency — {a.n} queries, generate={not a.no_generate}, "
              f"{time.strftime('%Y-%m-%d %H:%M')}",
              f"failed requests: {fails}",
              f"refusals: {refused or 'none'}"]
    report.append(table(f"COLD (first {a.cold} requests)", cold,
                        {"retrieval core": cold_core, "full pipeline": cold_tot}))
    report.append(table(f"WARM (remaining {len(warm_tot)})", warm,
                        {"retrieval core": warm_core, "full pipeline": warm_tot}))
    text = "\n".join(report)
    print("\n" + text)

    OUT.mkdir(parents=True, exist_ok=True)
    name = "latency_retrieval_only.txt" if a.no_generate else "latency_full_pipeline.txt"
    (OUT / name).write_text(text, encoding="utf-8")
    print(f"\nsaved -> {OUT / name}")


if __name__ == "__main__":
    main()
