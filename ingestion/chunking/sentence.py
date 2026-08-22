"""
Cutter 2 — SENTENCE WINDOW.

Cut the passage into single sentences. Search over the sentences.
But when we find one, hand the LLM the WHOLE passage it came from.

Why this usually wins:
  A question matches one exact sentence, not a whole paragraph. Searching
  sentence by sentence is sharper. But one sentence alone is too thin for
  the LLM to write a good answer from. So we search small and answer big.

Cost: more chunks means a bigger index. A 5-sentence passage becomes
5 rows instead of 1.
"""
from .base import Chunk
from .splitters import split_sentences


class SentenceWindowChunker:
    name = "sentence_window"

    def __init__(self, window: int = 1):
        # window=0 -> just the sentence. window=1 -> sentence plus one neighbour
        # on each side. Bigger window = more context, less sharpness.
        self.window = window

    def chunk(self, text: str, meta: dict) -> list[Chunk]:
        sents = split_sentences(text)
        out = []

        for i, s in enumerate(sents):
            lo = max(0, i - self.window)
            hi = min(len(sents), i + self.window + 1)
            window_text = " ".join(sents[lo:hi])

            out.append(Chunk(
                text=s,                       # search over this
                parent_text=text,             # answer from this
                meta={**meta,
                      "strategy": self.name,
                      "pos": i,
                      "n_sents": len(sents),
                      "window_text": window_text},
            ))

        return out