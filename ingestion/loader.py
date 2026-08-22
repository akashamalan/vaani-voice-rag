"""
One place where the dataset gets read. Both build_dataset.py and benchmark.py
import from here, so the extraction logic can never drift between them.

Two things that bit us and are now fixed here for good:

1. streaming=True does not work on this dataset.
   The parquet file is ONE row group of 778,638 rows, and `passages` is a
   struct-of-lists. pyarrow raises:
       ArrowNotImplementedError: Nested data conversions not implemented
                                 for chunked array outputs
   Fix: download the file once, then read it with pq.iter_batches.
   The download is a few GB and caches, so reruns are instant.

2. The real field names are Translated_passages / English_passages /
   is_selected. NOT passage_text. Reading the wrong key returns an empty
   list with no error, which produces a benchmark full of fake numbers.
"""
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "ai4bharat/MSMARCO-XI"

# Verified against list_repo_files(). Telugu has NO train file — validation only.
LANG_FILE = {
    "as": "train/asmtrain.parquet",   # Assamese
    "bn": "train/bentrain.parquet",   # Bengali
    "gu": "train/gujtrain.parquet",   # Gujarati
    "hi": "train/hintrain.parquet",   # Hindi
    "kn": "train/kantrain.parquet",   # Kannada
    "ml": "train/maltrain.parquet",   # Malayalam
    "mr": "train/martrain.parquet",   # Marathi
    "ne": "train/neptrain.parquet",   # Nepali
    "or": "train/oritrain.parquet",   # Odia
    "pa": "train/pantrain.parquet",   # Punjabi
    "sa": "train/santrain.parquet",   # Sanskrit
    "ta": "train/tamtrain.parquet",   # Tamil
    "ur": "train/urdtrain.parquet",   # Urdu
}


def get_parquet(lang: str) -> Path:
    """Download once, reuse forever. Returns the local path."""
    if lang not in LANG_FILE:
        raise ValueError(f"no train file for {lang!r}. have: {sorted(LANG_FILE)}")
    path = hf_hub_download(REPO, LANG_FILE[lang], repo_type="dataset")
    print(f"[file] {lang} -> {path}")
    return Path(path)


def iter_rows(lang: str, batch_size: int = 512, skip: int = 0):
    """
    Yields one row dict at a time: {"query": str, "passages": {...}}.
    skip = how many rows to jump over, for resuming a crashed run.
    """
    pf = pq.ParquetFile(get_parquet(lang))
    print(f"[file] {pf.metadata.num_rows:,} rows")

    seen = 0
    for batch in pf.iter_batches(batch_size=batch_size,
                                 columns=["query", "passages"]):
        rows = batch.to_pylist()
        if seen + len(rows) <= skip:      # whole batch already done
            seen += len(rows)
            continue
        for r in rows:
            if seen >= skip:
                yield r
            seen += 1


def extract_passages(row, use_english: bool = False):
    """
    Returns [(text, is_selected), ...].

    Real schema:
        passages: struct<
            English_passages:    list<string>
            Translated_passages: list<string>   <- the Indic text
            is_selected:         list<int64>
        >
    """
    p = row.get("passages") or {}

    key = "English_passages" if use_english else "Translated_passages"
    texts = p.get(key) or []
    sel = p.get("is_selected") or [0] * len(texts)

    out = []
    for t, s in zip(texts, sel):
        if t and t.strip():
            out.append((t.strip(), int(s)))
    return out
