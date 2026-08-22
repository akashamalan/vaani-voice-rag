"""
Pick a cutter by name from the command line:

    python scripts/build_dataset.py --strategy sentence_window
"""
from .base import Chunk, Chunker
from .passage import PassageChunker
from .sentence import SentenceWindowChunker
from .sliding_window import SlidingWindowChunker
from .semantic import SemanticChunker
from .metadata import MetadataAwareChunker

__all__ = ["Chunk", "Chunker", "get_chunker", "STRATEGIES"]

STRATEGIES = ["passage", "sentence_window", "sliding_window",
              "semantic", "metadata_aware"]


def get_chunker(name: str, encoder=None):
    if name == "passage":
        return PassageChunker()
    if name == "sentence_window":
        return SentenceWindowChunker(window=1)
    if name == "sliding_window":
        return SlidingWindowChunker(size=80, overlap=20)
    if name == "semantic":
        if encoder is None:
            raise ValueError("semantic chunker needs an encoder")
        return SemanticChunker(encoder)
    if name == "metadata_aware":
        return MetadataAwareChunker(SentenceWindowChunker(window=1))
    raise ValueError(f"unknown strategy {name!r}. pick from {STRATEGIES}")