"""
Cutter 3 — SLIDING WINDOW (overlap).

Cut into fixed-size pieces of words, but let each piece share some words
with the next one.

Why the overlap matters:
  Cut hard at 80 words and a fact sitting on the boundary gets sliced in
  half. Neither half makes sense, so neither half matches the question.
  Overlap means every fact appears whole in at least one piece.

size=80, overlap=20 means:
  piece 1 = words 0-80
  piece 2 = words 60-140
  piece 3 = words 120-200

Note: MS MARCO passages are short (~60-100 words). Many will come out as
one piece, same as the baseline. That is a real finding, not a bug — the
benchmark will show this cutter barely helps here, and saying so is worth
more than pretending it helped.
"""
from .base import Chunk


class SlidingWindowChunker:
    name = "sliding_window"

    def __init__(self, size: int = 80, overlap: int = 20):
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size = size
        self.overlap = overlap
        self.step = size - overlap

    def chunk(self, text: str, meta: dict) -> list[Chunk]:
        words = text.split()

        if len(words) <= self.size:
            return [Chunk(text, text,
                          {**meta, "strategy": self.name, "pos": 0, "n_parts": 1})]

        out, i, pos = [], 0, 0
        while i < len(words):
            piece = " ".join(words[i:i + self.size])
            out.append(Chunk(
                text=piece,
                parent_text=text,
                meta={**meta, "strategy": self.name, "pos": pos},
            ))
            if i + self.size >= len(words):
                break
            i += self.step
            pos += 1

        for c in out:
            c.meta["n_parts"] = len(out)
        return out