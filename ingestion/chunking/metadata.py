"""
Cutter 5 — METADATA AWARE.

This one does not cut. It wraps another cutter and adds a small tag to the
front of the text before we turn it into numbers.

Before:  "The Eiffel Tower is 330 metres tall."
After:   "[hi | numeric] The Eiffel Tower is 330 metres tall."

Why bother:
  A "how tall" question and a "who built" question are different kinds of
  question. Tagging the chunk lets a numeric question lean toward numeric
  chunks. And the language tag stops a Hindi question from matching a Tamil
  chunk that only looks similar because both are non-English.

The tag is added ONLY to the searched text. parent_text stays clean, so the
LLM never sees our tags.
"""
import re

from .base import Chunk

_HAS_NUM = re.compile(r'\d')
_YEAR = re.compile(r'\b(1[0-9]{3}|20[0-9]{2})\b')
_MONEY = re.compile(r'[$₹€£]|\b(rupees|dollars|crore|lakh)\b', re.I)


def content_tags(text: str) -> list[str]:
    tags = []
    if _YEAR.search(text):
        tags.append("date")
    if _MONEY.search(text):
        tags.append("money")
    if _HAS_NUM.search(text) and "date" not in tags and "money" not in tags:
        tags.append("numeric")
    if not tags:
        tags.append("text")
    return tags


class MetadataAwareChunker:
    name = "metadata_aware"

    def __init__(self, inner):
        # inner: any other cutter. Usually SentenceWindowChunker.
        self.inner = inner
        self.name = f"metadata_aware({inner.name})"

    def chunk(self, text: str, meta: dict) -> list[Chunk]:
        out = []
        for c in self.inner.chunk(text, meta):
            tags = content_tags(c.text)
            lang = c.meta.get("lang", "unk")
            prefix = f"[{lang} | {' '.join(tags)}] "

            out.append(Chunk(
                text=prefix + c.text,        # tagged — this gets embedded
                parent_text=c.parent_text,   # clean — this goes to the LLM
                meta={**c.meta, "strategy": self.name, "tags": tags},
            ))
        return out