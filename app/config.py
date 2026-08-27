"""
One place for every setting. Nothing below this file reads os.environ.

Keys are read from the environment and never from a file in the repo.
Set them with setx, open a NEW terminal, then start the server:

    setx SARVAM_API_KEY "..."
    setx GROQ_API_KEY   "..."
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- artifacts
ARTIFACTS = Path(os.environ.get("VOICE_RAG_ARTIFACTS",
                                r"C:\Users\akash\voice-rag-artifacts"))
LANG = os.environ.get("VOICE_RAG_LANG", "hi")

# ---------------------------------------------------------------- retrieval
EMBED_MODEL = "intfloat/multilingual-e5-small"
EMBED_DEVICE = os.environ.get("VOICE_RAG_DEVICE", "cpu")   # the VPS has no GPU
# efSearch 256 measured 0.610 recall@10 vs 0.590 at 64, for ~2ms. On the
# 500k build 64 was enough; at 1.485M the extra breadth pays for itself.
EF_SEARCH = 256
# k=5 measured a 65% miss rate on 20 qrels queries. recall@10 is 0.590 vs
# ~0.35 at k=5 — the correct passage is often at rank 5-9.
TOP_K = 10

# ---------------------------------------------------------------- upstreams
SARVAM_URL = "https://api.sarvam.ai/speech-to-text"
# saarika:v2 was deprecated — Sarvam returned an explicit error naming v2.5.
SARVAM_MODEL = "saarika:v2.5"
SARVAM_LANG = {"hi": "hi-IN", "ta": "ta-IN", "bn": "bn-IN", "te": "te-IN"}

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# llama-3.3-70b-versatile returned HTTP 404 — retired. Overridable without a
# code change, because model availability moves faster than deploys:
#   curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $KEY"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_MAX_TOKENS = 300
GROQ_TEMPERATURE = 0.2

HTTP_TIMEOUT = 30.0
RETRY_ATTEMPTS = 3

# ------------------------------------------------------------- cold start
# Windows trims an idle process's working set. Measured: the server fell to
# 153MB while the FAISS index alone is 2.6GB, and the next visitor paid
# 1794-2708ms in vector search while pages faulted back from the pagefile.
#
# Two mitigations, because neither is sufficient alone:
#   KEEPALIVE_SECONDS  periodic real retrieval. Keeps the model and the
#                      graph's upper layers hot — but ONE HNSW search only
#                      touches roughly efSearch*M nodes (~12MB of 2.6GB), so
#                      it cannot hold the whole index by itself.
#   PIN_WORKING_SET_GB SetProcessWorkingSetSizeEx with a HARD minimum, which
#                      is what actually forbids the trim. Verified settable
#                      without admin on this machine. Cost: that much RAM is
#                      reserved from everything else. Set 0 to disable.
KEEPALIVE_SECONDS = 180
PIN_WORKING_SET_GB = float(os.environ.get("VOICE_RAG_PIN_GB", "3.5"))

# ---------------------------------------------------------------- guardrails
# Off-topic detection is THREE layers, not one. Each catches what the others
# cannot, and the reason this is not a single tuned number is measured:
#
#   8 real questions   top1 0.8689-0.9470   margin 0.0125-0.0419
#   6 junk questions   top1 0.8174-0.8698   margin 0.0054-0.0166
#
# The ranges overlap. "purple monkey dishwasher velocity" scores 0.8174.
# e5 embeddings occupy a narrow cone, so every query — meaningful or not —
# lands between roughly 0.81 and 0.95. The number does not encode meaning
# finely enough to threshold on, and any boundary that separates a small
# sample is fitting noise.
#
#   layer 1  MIN_TOP_SCORE   cheap floor. Skips the LLM on empty retrievals
#                            only. It is EXPECTED to catch nothing in the
#                            junk set above. Raising it does not help; it
#                            just starts refusing real questions.
#   layer 2  NOT_IN_CONTEXT  the actual off-topic guardrail. The model reads
#                            the real passages and the real question, which
#                            is semantic judgment no cosine can do. Costs
#                            nothing extra — it is the call we already make.
#   layer 3  grounding       catches fabrication after generation.
MIN_TOP_SCORE = 0.80     # floor only. See above before changing this.

# Grounding is content-word overlap, NOT raw overlap. Measured: with raw
# overlap a fabricated answer scored 0.600 against a threshold of 0.25 and
# passed, because की/है/राजधानी match almost any passage. Function words are
# removed before scoring, which pushed fabrications to 0.33-0.67 and left
# correct answers at 1.00.
MIN_GROUNDING = 0.8


MIN_RARE_LEN = 4

NOT_IN_CONTEXT = "NOT_IN_CONTEXT"


def require(name: str) -> str:
    """Fail loudly at startup rather than quietly at the first user request."""
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(
            f"{name} is not set. Run:  setx {name} \"your-key\"  "
            f"then open a NEW terminal (setx does not affect the current one)."
        )
    return v


def sarvam_key() -> str:
    return require("SARVAM_API_KEY")


def groq_key() -> str:
    return require("GROQ_API_KEY")
