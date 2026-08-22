"""
The wire contract. The frontend reads nothing that is not defined here.

Every websocket message is a dict with a "type". Every path through the
pipeline ends in exactly one of:
    refusal + done
    tokens  + done
    error
There is no fourth option and no silent exit.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field

# ------------------------------------------------------------------ stages
Stage = Literal["stt", "safety", "embed", "search", "fetch",
                "relevance", "generate", "grounding"]

# ----------------------------------------------------------- refusal reasons
UNSAFE_INPUT = "unsafe_input"
OFF_TOPIC = "off_topic"
NO_RELEVANT_CONTEXT = "no_relevant_context"
ANSWER_NOT_GROUNDED = "answer_not_grounded"
EMPTY_QUERY = "empty_query"
UPSTREAM_ERROR = "upstream_error"

REFUSAL_MESSAGE = {
    UNSAFE_INPUT: "I can't help with that request.",
    OFF_TOPIC: "That's outside what I can answer from my sources.",
    NO_RELEVANT_CONTEXT: "I couldn't find anything relevant to that.",
    ANSWER_NOT_GROUNDED: "I found passages but couldn't answer from them reliably.",
    EMPTY_QUERY: "I didn't catch a question.",
    UPSTREAM_ERROR: "Something upstream failed. Please try again.",
}


# ------------------------------------------------------------- HTTP schemas
class QueryRequest(BaseModel):
    text: str
    k: int = Field(default=5, ge=1, le=20)
    # generate=False -> retrieval only, no Groq call. Used for threshold
    # tuning and as the latency harness for the retrieval core alone.
    generate: bool = True


class PassageOut(BaseModel):
    doc_id: str
    text: str
    score: float
    rank: int


class QueryResponse(BaseModel):
    query: str
    answer: Optional[str] = None
    refused: bool = False
    refusal_reason: Optional[str] = None
    passages: list[PassageOut] = []
    timings_ms: dict[str, float] = {}
    total_ms: float = 0.0
    top_score: float = 0.0
    grounding_score: Optional[float] = None


# ---------------------------------------------------------- socket events
def ev_stage(stage: str, ms: float, detail: Optional[str] = None) -> dict:
    e = {"type": "stage", "stage": stage, "ms": round(ms, 2)}
    if detail:
        e["detail"] = detail
    return e


def ev_transcript(text: str) -> dict:
    return {"type": "transcript", "text": text}


def ev_passages(passages: list[dict]) -> dict:
    return {"type": "passages", "passages": passages}


def ev_token(text: str) -> dict:
    return {"type": "token", "text": text}


def ev_refusal(reason: str, message: Optional[str] = None) -> dict:
    return {"type": "refusal", "reason": reason,
            "message": message or REFUSAL_MESSAGE.get(reason, "I can't answer that.")}


def ev_done(timings_ms: dict, total_ms: float,
            grounding_score: Optional[float] = None) -> dict:
    e = {"type": "done",
         "timings_ms": {k: round(v, 2) for k, v in timings_ms.items()},
         "total_ms": round(total_ms, 2)}
    if grounding_score is not None:
        e["grounding_score"] = round(grounding_score, 4)
    return e


def ev_error(message: str) -> dict:
    return {"type": "error", "message": message}
