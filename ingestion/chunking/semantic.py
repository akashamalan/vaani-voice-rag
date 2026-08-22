"""
Cutter 4 — SEMANTIC.

Cut where the meaning changes, not at a fixed word count.

How it works:
  1. Split into sentences.
  2. Turn each sentence into numbers.
  3. Compare each sentence to the next one.
  4. Where two neighbours are very different, that is a topic change. Cut there.
  5. Sentences that stay similar get glued into one chunk.

Honest warning:
  This is the slow one. It runs the embedding model at cutting time, not
  just at indexing time. And MS MARCO passages are short and usually about
  one thing, so there is often nothing to cut. Expect this to score close
  to the baseline while costing much more.

  Say that in your writeup. "We tried it, measured it, it did not help on
  this dataset, here are the numbers" is a stronger answer than pretending
  everything worked.
"""
import numpy as np

from .base import Chunk
from .splitters import split_sentences


class SemanticChunker:
    name = "semantic"

    def __init__(self, encoder, percentile: int = 80,
                 min_sents: int = 3, max_words: int = 200):
        self.encoder = encoder
        self.percentile = percentile
        self.min_sents = min_sents
        self.max_words = max_words

    def chunk(self, text: str, meta: dict) -> list[Chunk]:
        sents = split_sentences(text)

        if len(sents) < self.min_sents:
            return [Chunk(text, text,
                          {**meta, "strategy": self.name, "pos": 0, "n_parts": 1})]

        vecs = self.encoder.encode(sents)          
        vecs = np.asarray(vecs, dtype=np.float32)

        sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
        distances = 1.0 - sims 

   
        threshold = np.percentile(distances, self.percentile)
        cut_after = {i for i, d in enumerate(distances) if d >= threshold}

        groups, current = [], []
        for i, s in enumerate(sents):
            current.append(s)
            too_long = len(" ".join(current).split()) >= self.max_words
            if i in cut_after or too_long:
                groups.append(current)
                current = []
        if current:
            groups.append(current)

        out = []
        for pos, g in enumerate(groups):
            piece = " ".join(g)
            out.append(Chunk(
                text=piece,
                parent_text=text,
                meta={**meta, "strategy": self.name, "pos": pos,
                      "n_parts": len(groups)},
            ))
        return out