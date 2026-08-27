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
import asyncio
import ctypes
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import (ARTIFACTS, EF_SEARCH, EMBED_DEVICE, KEEPALIVE_SECONDS,
                        LANG, PIN_WORKING_SET_GB, TOP_K)
from app.generation.generator import GroqGenerator
from app.harness.pipeline import run_pipeline
from app.harness.schemas import (PassageOut, QueryRequest, QueryResponse,
                                 ev_error, ev_stage, ev_transcript)
from app.retrieval.vector_store import VectorStore
from app.stt.sarvam import SarvamSTT

STATE: dict = {"store": None, "gen": None, "stt": None, "missing": []}

# Deliberately varied: HNSW visits only the nodes along its search path, so
# repeating one query would keep re-touching the same ~12MB of the graph.
# Different topics enter the graph at different points and hold more of it.
KEEPALIVE_QUERIES = [
    "भारत की राजधानी क्या है",
    "कंप्यूटर सॉफ्टवेयर कैसे काम करता है",
    "स्वास्थ्य और पोषण के लाभ",
    "इतिहास में प्रसिद्ध युद्ध",
    "बैंक ऋण की ब्याज दर",
]


def _pin_working_set(gb: float) -> str:
    """
    Forbid Windows from trimming the working set below `gb`.

    This is the mitigation that actually works. The keepalive keeps the hot
    path warm; only a hard minimum stops the OS reclaiming the rest of the
    2.6GB index while the process is idle. Verified settable without admin.
    """
    if gb <= 0:
        return "disabled"
    if sys.platform != "win32":
        return "skipped (not windows)"
    try:
        import ctypes.wintypes as w
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = w.HANDLE
        set_ex = k32.SetProcessWorkingSetSizeEx
        set_ex.argtypes = [w.HANDLE, ctypes.c_size_t, ctypes.c_size_t, w.DWORD]
        HARD_MIN, NO_HARD_MAX = 0x00000001, 0x00000008
        lo = int(gb * 1024 ** 3)
        ok = set_ex(k32.GetCurrentProcess(), lo, int(lo * 1.5),
                    HARD_MIN | NO_HARD_MAX)
        if not ok:
            return f"FAILED (winerr {ctypes.get_last_error()})"
        return f"min {gb:.1f}GB pinned"
    except Exception as exc:                                  # noqa: BLE001
        return f"unavailable ({type(exc).__name__}: {exc})"


async def _keepalive():
    """Real retrieval on a timer. No Groq — retrieval only, no quota spent."""
    i = 0
    while True:
        await asyncio.sleep(KEEPALIVE_SECONDS)
        store = STATE.get("store")
        if store is None:
            continue
        q = KEEPALIVE_QUERIES[i % len(KEEPALIVE_QUERIES)]
        i += 1
        try:
            t0 = perf_counter()
            await asyncio.to_thread(store.search, q, 5)
            print(f"[keepalive] {(perf_counter()-t0)*1000:.1f} ms")
        except asyncio.CancelledError:
            raise
        except Exception as exc:                              # noqa: BLE001
            print(f"[keepalive] failed: {type(exc).__name__}: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # before load, so the index is faulted into an already-protected set
    print(f"[workingset] {_pin_working_set(PIN_WORKING_SET_GB)}")

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

    ka = asyncio.create_task(_keepalive())
    print(f"[keepalive] every {KEEPALIVE_SECONDS}s, retrieval only")

    yield

    ka.cancel()
    try:
        await ka
    except asyncio.CancelledError:
        pass

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


# ---------------------------------------------------------------- frontend
# MUST be registered last. Starlette matches routes in registration order, so
# a mount at "/" declared earlier would swallow /health, /api/query and
# /ws/query before they were ever reached.
#
# Serving the built SPA from the same origin means one tunnel, one URL, and no
# CORS. Build it first:  npm run build --prefix frontend
_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if (_DIST / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")
    print(f"[static] serving {_DIST}")
else:
    print(f"[static] {_DIST} not built — root will 404. "
          f"Run: npm run build --prefix frontend")
