"""
Splitting sentences in Indic text.

English ends sentences with .  ?  !
Hindi, Marathi, Bengali end with  ।  (called a danda)
Some text uses  ॥  (double danda)

A plain .split(".") loses every Hindi sentence. This does not.
"""
import re

_END = re.compile(r'(?<=[।॥.!?])\s+')
_ABBREV = re.compile(r'\b(Mr|Mrs|Dr|St|Jr|Sr|vs|etc|No|Inc|Ltd)\.$', re.I)


def split_sentences(text: str, min_chars: int = 15) -> list[str]:
    parts = [p.strip() for p in _END.split(text) if p.strip()]

    # glue back pieces that are too short to stand alone, and abbreviations
    out: list[str] = []
    for p in parts:
        if out and (len(p) < min_chars or _ABBREV.search(out[-1])):
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)

    return out or ([text.strip()] if text.strip() else [])


def word_count(text: str) -> int:
    return len(text.split())