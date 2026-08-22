"""
Cutter 1 — WHOLE PASSAGE. The baseline.

Does nothing. One passage in, one chunk out.

This is here on purpose. Every other cutter has to beat this number,
or it was not worth the extra work. Without a baseline the comparison
table means nothing.
"""
from .base import Chunk


class PassageChunker:
    name = "passage"

    def chunk(self, text: str, meta: dict) -> list[Chunk]:
        return [Chunk(
            text=text,
            parent_text=text,
            meta={**meta, "strategy": self.name, "pos": 0},
        )]