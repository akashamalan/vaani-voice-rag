"""
Reciprocal Rank Fusion of dense and lexical results.

RRF is used instead of score blending because the two scales are not
comparable and one of them is nearly useless as a magnitude: dense cosines
here occupy a narrow 0.856-0.956 band across both hits and misses, while BM25
scores are unbounded and corpus-dependent. Any weighted sum would need
normalization that the dense side cannot support. RRF reads only rank
position, so it sidesteps the problem entirely.

    score(d) = sum over lists of 1 / (k + rank(d))

k=60 is the value from the original Cormack et al. formulation. Larger k
flattens the contribution of top ranks; smaller k lets a single list dominate.
"""
from collections import defaultdict

RRF_K = 60


def rrf(*ranked_lists: list[str], k: int = RRF_K) -> list[tuple[str, float]]:
    """
    Each argument is a list of doc_ids in rank order (best first).
    Returns [(doc_id, fused_score), ...] sorted best first.
    """
    scores: dict[str, float] = defaultdict(float)
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def fuse(dense_ids: list[str], lexical_ids: list[str], k: int = RRF_K,
         top_k: int = 10) -> list[str]:
    """Convenience wrapper returning just the fused doc_id ordering."""
    return [d for d, _ in rrf(dense_ids, lexical_ids, k=k)[:top_k]]
