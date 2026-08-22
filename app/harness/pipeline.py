"""
The pipeline. One async generator of events, consumed by both transports:
the websocket forwards each event as it is produced, /api/query collects
them into a single response. Same code path, so the two can never disagree.

Stage order is safety -> retrieval -> relevance -> generate -> grounding,
and it is ordered by cost. Safety is a regex, so it runs before we spend
40ms embedding. Relevance is a float compare, so it runs before we spend
hundreds of milliseconds in Groq. We never call the LLM for a question that
has already failed.
"""
import asyncio
from time import perf_counter

from app.config import NOT_IN_CONTEXT, TOP_K
from app.guardrails.checks import (check_grounding, check_relevance,
                                   check_safety)
from app.harness.retry import UpstreamError
from app.harness.schemas import (ANSWER_NOT_GROUNDED, NO_RELEVANT_CONTEXT,
                                 OFF_TOPIC, UPSTREAM_ERROR, ev_done,
                                 ev_error, ev_passages, ev_refusal, ev_stage,
                                 ev_token)


async def run_pipeline(store, generator, question: str, k: int = TOP_K,
                       generate: bool = True):
    """
    Yields wire events. Always terminates in refusal+done, tokens+done, or error.

    generate=False stops after the relevance check and returns passages,
    top_score and timings without calling Groq. That is how you collect the
    score distribution needed to place MIN_TOP_SCORE, and it works with no
    GROQ_API_KEY set.
    """
    timings: dict[str, float] = {}
    t_start = perf_counter()
    total = lambda: (perf_counter() - t_start) * 1000        # noqa: E731

    try:
        # ---------------------------------------------------------- safety
        t0 = perf_counter()
        ok, reason = check_safety(question)
        timings["safety"] = (perf_counter() - t0) * 1000
        yield ev_stage("safety", timings["safety"])
        if not ok:
            yield ev_refusal(reason)
            yield ev_done(timings, total())
            return

        # ------------------------------------------------------- retrieval
        # search() is synchronous and CPU-bound (~41ms). Off the event loop,
        # or one request blocks every other connection.
        result = await asyncio.to_thread(store.search, question, k)
        for stage in ("embed", "search", "fetch"):
            if stage in result.timings_ms:
                timings[stage] = result.timings_ms[stage]
                yield ev_stage(stage, timings[stage])

        passages = [{"doc_id": h.doc_id, "text": h.text,
                     "score": round(h.score, 4), "rank": h.rank}
                    for h in result.hits]
        yield ev_passages(passages)

        # ------------------------------------------------------- relevance
        t0 = perf_counter()
        relevant = check_relevance(result.top_score)
        timings["relevance"] = (perf_counter() - t0) * 1000
        yield ev_stage("relevance", timings["relevance"],
                       f"top_score {result.top_score:.3f}")
        if not relevant:
            yield ev_refusal(NO_RELEVANT_CONTEXT)
            yield ev_done(timings, total())
            return

        # retrieval-only: everything above is measured, nothing below costs money
        if not generate:
            yield ev_done(timings, total())
            return

        # -------------------------------------------------------- generate
        texts = [h.text for h in result.hits]
        t0 = perf_counter()
        first_token = None
        emitted, held = [], ""
        streaming = False

        async for piece in generator.stream(question, texts):
            if first_token is None:
                first_token = (perf_counter() - t0) * 1000
                timings["generate"] = first_token
                yield ev_stage("generate", first_token, "time to first token")

            if streaming:
                emitted.append(piece)
                yield ev_token(piece)
                continue

            # Hold the opening characters until we know this is not the
            # NOT_IN_CONTEXT sentinel. Leaking that string onto the screen
            # during a demo would be worse than a few ms of delay.
            held += piece
            if NOT_IN_CONTEXT.startswith(held.strip()):
                continue
            streaming = True
            emitted.append(held)
            yield ev_token(held)
            held = ""

        answer = ("".join(emitted) + held).strip()

        if first_token is None:                     # stream produced nothing
            yield ev_refusal(UPSTREAM_ERROR)
            yield ev_done(timings, total())
            return

        # the model declining is a refusal, and the desired behaviour
        if NOT_IN_CONTEXT in answer:
            yield ev_refusal(OFF_TOPIC)
            yield ev_done(timings, total())
            return

        if held:                                    # short answer, never flushed
            yield ev_token(held)

        # ------------------------------------------------------- grounding
        t0 = perf_counter()
        grounded, score, unsupported = check_grounding(answer, texts)
        timings["grounding"] = (perf_counter() - t0) * 1000
        detail = f"overlap {score:.3f}"
        if unsupported:
            detail += f" | unsupported: {', '.join(unsupported[:5])}"
        yield ev_stage("grounding", timings["grounding"], detail)
        if not grounded:
            # tokens are already on screen — the UI should mark the answer
            # unreliable rather than pretend it was never shown
            yield ev_refusal(ANSWER_NOT_GROUNDED)

        yield ev_done(timings, total(), score)

    except UpstreamError as exc:
        yield ev_refusal(UPSTREAM_ERROR, str(exc))
        yield ev_done(timings, total())
    except Exception as exc:                                  # noqa: BLE001
        yield ev_error(f"{type(exc).__name__}: {exc}")
