"""
One shape for every cutter.

  text        -> what we turn into numbers and search over
  parent_text -> what we hand to the LLM after we find a match
  meta        -> lang, which passage it came from, which cutter made it

text and parent_text are often different on purpose. We search with a small
precise piece, then answer with the bigger piece around it.
"""
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Chunk:
    text: str
    parent_text: str
    meta: dict = field(default_factory=dict)


class Chunker(Protocol):
    name: str

    def chunk(self, text: str, meta: dict) -> list[Chunk]:
        ...