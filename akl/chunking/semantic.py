"""Semantic boundary detection (PRD §4.2 pass 2).

Given sentence spans, computes sentence embeddings with an injected ``embed``
callable and marks a boundary *before* sentence ``i`` when the cosine similarity
between the running chunk centroid and sentence ``i`` drops below ``1 - threshold``.
When no embedder is configured the pass is a no-op (token pass alone applies).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def semantic_boundaries(
    sentences: Sequence[str], embed: Embedder | None, threshold: float, *, min_run: int = 2
) -> set[int]:
    """Indices ``i`` such that a new chunk should start at sentence ``i``."""
    if embed is None or len(sentences) < 2 * min_run:
        return set()
    vectors = [list(v) for v in embed(list(sentences))]
    boundaries: set[int] = set()
    centroid = list(vectors[0])
    run = 1
    for i in range(1, len(vectors)):
        sim = _cosine(centroid, vectors[i])
        if run >= min_run and sim < 1.0 - threshold:
            boundaries.add(i)
            centroid = list(vectors[i])
            run = 1
            continue
        centroid = [(c * run + v) / (run + 1) for c, v in zip(centroid, vectors[i], strict=False)]
        run += 1
    return boundaries
