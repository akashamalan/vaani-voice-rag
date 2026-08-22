"""
Three guardrails, none of which call an LLM.

An LLM classifier costs 200-400ms per check and would triple the latency
budget to catch cases a regex and a float compare already catch. These run
in microseconds.

    safety     regex, on the raw question, before anything expensive
    relevance  one float compare against the retriever's top score
    grounding  content-word overlap PLUS a verbatim check on numbers and
               rare tokens

Grounding is two tests because one is not enough:

  overlap   catches wholesale fabrication. "पेरिस फ्रांस की राजधानी है"
            against a Delhi passage scores 0.33 on content words.
  verbatim  catches entity substitution. "मुंबई भारत की राजधानी है" is 4/5
            grounded — the ratio barely moves — but मुंबई appears in no
            passage, so it fails outright.

An embedding check would be worse than either: "Mumbai is the capital" and
"Delhi is the capital" sit next to each other in vector space, which is
exactly the confusion we need to catch.
"""
import re

from app.config import MIN_GROUNDING, MIN_RARE_LEN, MIN_TOP_SCORE

# Devanagari explicitly — \w would mostly cover it, but being explicit means
# the behaviour does not change if the regex flags ever do.
_TOKEN = re.compile(r"[0-9A-Za-zऀ-ॿ]+")
_NUMERIC = re.compile(r"^[0-9]+$")

# Function words carry no grounding signal and dominate short answers.
# Hindi + English, deliberately small — this is a noise filter, not a
# linguistic resource.
STOPWORDS = set("""
की का के को में से पर और या यह वह जो कि ने भी एक है हैं था थे थी हो होता होती होने
तक ही तो लिए करने किया कर रहा रहे रही अपने इस उस कोई कुछ सब बहुत नहीं
the a an is are was were be been being in on at of and or to for that this these those
it its as by with from not no any some all more most very there their they them he she
""".split())

_HARM = re.compile(
    r"\b(how to (make|build|synthesi[sz]e) (a )?(bomb|explosive|weapon|poison|nerve agent)"
    r"|kill (myself|yourself|him|her|them)"
    r"|commit suicide"
    r"|child (porn|abuse|sexual)"
    r"|(buy|sell|make) (meth|heroin|fentanyl)"
    r")\b",
    re.I,
)

# Prompt injection. The question is user text that ends up next to a system
# prompt, so anything addressing the model as an operator is refused.
_INJECT = re.compile(
    r"(ignore (all )?(previous|prior|above) instructions"
    r"|disregard (the )?(system|previous) (prompt|instructions)"
    r"|you are now a\b"
    r"|reveal (your )?(system )?prompt"
    r"|show me your (system )?(prompt|instructions)"
    r"|repeat (the )?(text )?above"
    r"|act as (if you are|an?) "
    r")",
    re.I,
)


# Formatting the model adds that is not content. Stripped before grounding
# so a correct answer is not refused for saying "उत्तर:" or "**दिल्ली**".
_MD = re.compile(r"[*_`~]+")
_LEAD_LABEL = re.compile(r"^\s*([0-9A-Za-zऀ-ॿ]{1,20})\s*[:：]\s*")

# Only these words are stripped as labels — NOT any word before a colon.
# Stripping arbitrary leading words would let a hallucinated entity hide in
# the label position: "मुंबई: भारत की राजधानी है" would silently lose the
# one token the verbatim check exists to catch.
_LABEL_WORDS = {"उत्तर", "जवाब", "उत्तर", "answer", "ans", "note", "reply",
                "response", "output", "a"}


def normalize_answer(text: str) -> str:
    """Remove model formatting before scoring. Display text is unaffected."""
    t = _MD.sub("", text or "")
    m = _LEAD_LABEL.match(t)
    if m and m.group(1).lower() in _LABEL_WORDS:
        t = t[m.end():]
    return t.strip()


def tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text or "")}


def content_tokens(text: str) -> set[str]:
    return {t for t in tokens(text) if t not in STOPWORDS}


def check_safety(text: str) -> tuple[bool, str | None]:
    """(ok, reason). Runs before retrieval so unsafe input costs nothing."""
    from app.harness.schemas import EMPTY_QUERY, UNSAFE_INPUT
    if not text or not text.strip():
        return False, EMPTY_QUERY
    if _HARM.search(text) or _INJECT.search(text):
        return False, UNSAFE_INPUT
    return True, None


def check_relevance(top_score: float) -> bool:
    """One compare. If the best passage is weak, there is nothing to answer from."""
    return top_score >= MIN_TOP_SCORE


def grounding_score(answer: str, passages: list[str]) -> float:
    """Fraction of the answer's CONTENT words that appear in the passages."""
    a = content_tokens(answer)
    if not a:
        return 0.0
    ctx = set()
    for p in passages:
        ctx |= content_tokens(p)
    return len(a & ctx) / len(a)


def verbatim_violations(answer: str, passages: list[str]) -> list[str]:
    """
    Numbers and rare tokens that appear in the answer but in no passage.

    Any hit here is a hard fail regardless of the overlap ratio — this is
    the check that catches a swapped entity or an invented figure.
    """
    ctx = set()
    for p in passages:
        ctx |= tokens(p)

    bad = []
    for t in tokens(answer):
        if t in ctx:
            continue
        if _NUMERIC.match(t) or (t not in STOPWORDS and len(t) >= MIN_RARE_LEN):
            bad.append(t)
    return sorted(bad)


def check_grounding(answer: str, passages: list[str]) -> tuple[bool, float, list[str]]:
    """(ok, overlap_score, violations). Both tests must pass."""
    clean = normalize_answer(answer)
    score = grounding_score(clean, passages)
    bad = verbatim_violations(clean, passages)
    return (score >= MIN_GROUNDING and not bad), score, bad
