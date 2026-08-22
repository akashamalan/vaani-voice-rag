"""
FastAPI entrypoint.

    uvicorn app.main:app --reload --port 8000

Everything expensive is built once in the lifespan and reused: the 1.485M
vector index (24s to load), the embedding model, and one pooled httpx client
per upstream. Building any of these per request would cost more than the
entire latency budget.

The server starts even when SARVAM_API_KEY or GROQ_API_KEY are missing, so
retrieval can be tested in isolation. Each endpoint reports clearly which
key it needs rather than failing somewhere deep in a stack trace.
"""
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.config import ARTIFACTS, EF_SEARCH, EMBED_DEVICE, LANG, TOP_K
from app.generation.generator import GroqGenerator
from app.harness.pipeline import run_pipeline
from app.harness.schemas import (PassageOut, QueryRequest, QueryResponse,
                                 ev_error, ev_stage, ev_transcript)
from app.retrieval.vector_store import VectorStore
from app.stt.sarvam import SarvamSTT

STATE: dict = {"store": None, "gen": None, "stt": None, "missing": []}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["store"] = VectorStore(lang=LANG, artifacts=ARTIFACTS,
                                 ef_search=EF_SEARCH,
                                 device=EMBED_DEVICE).load()

    for name, build in (("GROQ_API_KEY", _build_gen),
                        ("SARVAM_API_KEY", _build_stt)):
        try:
            build()
        except RuntimeError as exc:
            STATE["missing"].append(name)
            print(f"[warn] {exc}")

    yield

    for key in ("gen", "stt"):
        obj = STATE.get(key)
        if obj is not None:
            await obj.client.aclose()
    if STATE["store"]:
        STATE["store"].close()


def _build_gen():
    client = GroqGenerator.make_client()
    STATE["gen"] = GroqGenerator(client)


def _build_stt():
    client = SarvamSTT.make_client()
    STATE["stt"] = SarvamSTT(client, lang=LANG)


app = FastAPI(title="voice-rag", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/health")
async def health():
    store = STATE["store"]
    return {
        "ok": store is not None,
        "lang": LANG,
        "vectors": store.index.ntotal if store else 0,
        "ef_search": EF_SEARCH,
        "groq": STATE["gen"] is not None,
        "sarvam": STATE["stt"] is not None,
        "missing_keys": STATE["missing"],
    }


@app.post("/api/query", response_model=QueryResponse)
async def api_query(req: QueryRequest):
    """
    Text in, one JSON out. This is also the latency benchmark harness —
    loop questions through it and collect timings_ms for P50/P70/P100.
    """
    if req.generate and STATE["gen"] is None:
        return QueryResponse(query=req.text, refused=True,
                             refusal_reason="upstream_error",
                             answer="GROQ_API_KEY is not set on the server. "
                                    "Use {\"generate\": false} for retrieval only.")

    out = QueryResponse(query=req.text)
    tokens: list[str] = []

    async for ev in run_pipeline(STATE["store"], STATE["gen"], req.text,
                                 req.k, generate=req.generate):
        t = ev["type"]
        if t == "token":
            tokens.append(ev["text"])
        elif t == "passages":
            out.passages = [PassageOut(**p) for p in ev["passages"]]
            out.top_score = out.passages[0].score if out.passages else 0.0
        elif t == "refusal":
            out.refused = True
            out.refusal_reason = ev["reason"]
            if not tokens:
                out.answer = ev["message"]
        elif t == "done":
            out.timings_ms = ev["timings_ms"]
            out.total_ms = ev["total_ms"]
            out.grounding_score = ev.get("grounding_score")
        elif t == "error":
            out.refused = True
            out.refusal_reason = "upstream_error"
            out.answer = ev["message"]

    if tokens:
        out.answer = "".join(tokens).strip()
    return out


@app.websocket("/ws/query")
async def ws_query(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            kind = msg.get("type")

            if kind == "audio":
                if STATE["stt"] is None:
                    await ws.send_json(ev_error("SARVAM_API_KEY is not set."))
                    continue
                from time import perf_counter
                t0 = perf_counter()
                try:
                    text = await STATE["stt"].transcribe(msg.get("data", ""))
                except Exception as exc:                      # noqa: BLE001
                    await ws.send_json(ev_error(f"stt failed: {exc}"))
                    continue
                await ws.send_json(ev_stage("stt", (perf_counter() - t0) * 1000))
                await ws.send_json(ev_transcript(text))
            elif kind == "text":
                text = msg.get("text", "")
            else:
                await ws.send_json(ev_error(f"unknown message type {kind!r}"))
                continue

            gen_wanted = msg.get("generate", True)
            if gen_wanted and STATE["gen"] is None:
                await ws.send_json(ev_error("GROQ_API_KEY is not set."))
                continue

            async for ev in run_pipeline(STATE["store"], STATE["gen"], text,
                                         msg.get("k", TOP_K),
                                         generate=gen_wanted):
                await ws.send_json(ev)

    except WebSocketDisconnect:
        return
    except Exception as exc:                                  # noqa: BLE001
        try:
            await ws.send_json(ev_error(f"{type(exc).__name__}: {exc}"))
        except Exception:                                     # noqa: BLE001
            pass
