"""
Prove the websocket contract end to end.

Sends a real question and an off-topic one, records every event, and checks
the invariants the frontend depends on:

    - every event type is one of the seven in the spec
    - every stage name is one of the eight
    - exactly one `done`, and it is last
    - the run ends in refusal+done, tokens+done, or error — never silently
    - the `generate` stage carries time to FIRST token, not total

Usage (server must already be running):
    python -u scripts/ws_probe.py
    python -u scripts/ws_probe.py --url ws://localhost:8000/ws/query
"""
import argparse
import asyncio
import json
import sys

import websockets

EVENTS = {"stage", "transcript", "passages", "token", "refusal", "done", "error"}
STAGES = {"stt", "safety", "embed", "search", "fetch",
          "relevance", "generate", "grounding"}


async def probe(url: str, payload: dict, label: str) -> bool:
    print(f"\n{'='*72}\n{label}\n  send: {json.dumps(payload, ensure_ascii=False)}\n{'='*72}")
    events, tokens = [], []
    try:
        async with websockets.connect(url, open_timeout=15) as ws:
            await ws.send(json.dumps(payload))
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=90)
                ev = json.loads(raw)
                events.append(ev)
                t = ev.get("type")
                if t == "stage":
                    d = f"  ({ev['detail']})" if ev.get("detail") else ""
                    print(f"  stage      {ev.get('stage'):<10} {ev.get('ms'):>8} ms{d}")
                elif t == "transcript":
                    print(f"  transcript {ev.get('text','')[:60]}")
                elif t == "passages":
                    ps = ev.get("passages", [])
                    print(f"  passages   {len(ps)} hits, top score "
                          f"{ps[0]['score'] if ps else 'n/a'}")
                elif t == "token":
                    tokens.append(ev.get("text", ""))
                elif t == "refusal":
                    print(f"  REFUSAL    {ev.get('reason')} — {ev.get('message')}")
                elif t == "error":
                    print(f"  ERROR      {ev.get('message')}")
                    break
                elif t == "done":
                    print(f"  done       total {ev.get('total_ms')} ms  "
                          f"grounding {ev.get('grounding_score')}")
                    break
    except Exception as exc:                                   # noqa: BLE001
        print(f"  !! connection/protocol failure: {type(exc).__name__}: {exc}")
        return False

    if tokens:
        print(f"  answer     {''.join(tokens).strip()[:200]}")

    # ------------------------------------------------------------ invariants
    ok = True
    kinds = [e.get("type") for e in events]

    bad = {k for k in kinds if k not in EVENTS}
    ok &= not bad
    print(f"\n  [{'PASS' if not bad else 'FAIL'}] event types valid"
          + (f" — unexpected {bad}" if bad else ""))

    seen_stages = [e["stage"] for e in events if e.get("type") == "stage"]
    bad_s = {s for s in seen_stages if s not in STAGES}
    ok &= not bad_s
    print(f"  [{'PASS' if not bad_s else 'FAIL'}] stage names valid "
          f"— saw {seen_stages}")

    n_done = kinds.count("done")
    done_last = kinds and kinds[-1] in ("done", "error")
    ok &= (n_done == 1 or kinds[-1] == "error") and done_last
    print(f"  [{'PASS' if done_last else 'FAIL'}] terminates on done/error "
          f"(done x{n_done}, last={kinds[-1] if kinds else None})")

    terminal = ("refusal" in kinds and "done" in kinds) or \
               (tokens and "done" in kinds) or ("error" in kinds)
    ok &= bool(terminal)
    print(f"  [{'PASS' if terminal else 'FAIL'}] ends refusal+done / tokens+done / error")

    gen = [e for e in events if e.get("type") == "stage" and e["stage"] == "generate"]
    if gen:
        d = gen[0].get("detail", "")
        good = "first token" in d.lower()
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] generate.ms is time to first token "
              f"— detail={d!r}")
    else:
        print("  [ -- ] no generate stage (expected when refused before the LLM)")

    return ok


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8000/ws/query")
    a = ap.parse_args()

    results = []
    results.append(await probe(
        a.url, {"type": "text", "text": "भारत की राजधानी क्या है"},
        "REAL QUESTION — expect passages, tokens, grounding, done"))
    results.append(await probe(
        a.url, {"type": "text", "text": "purple monkey dishwasher velocity"},
        "OFF-TOPIC — expect refusal (off_topic via NOT_IN_CONTEXT) + done"))
    results.append(await probe(
        a.url, {"type": "text", "text": "भारत की राजधानी क्या है", "generate": False},
        "RETRIEVAL ONLY — expect passages + done, no tokens, no Groq"))

    print(f"\n{'='*72}")
    print("OVERALL:", "ALL CONTRACT CHECKS PASSED" if all(results)
          else "FAILURES ABOVE — contract not satisfied")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
