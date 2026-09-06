"""Threshold calibration (PRD §12.5): sweep ``rag_min_confidence``/``rag_strong_confidence`` and
report how retrieval sufficiency and refusal precision/recall respond, to help pick values.

This does not re-run retrieval per threshold (expensive); it re-evaluates the *gate* against
already-computed per-query confidence/expected-answerability pairs from one eval run, using the
exact gate logic in :mod:`akl.rag.retrieval.engine` so the sweep matches production behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from akl.eval.metrics import RefusalCounts


@dataclass(frozen=True)
class ThresholdPoint:
    min_confidence: float
    refusal_precision: float
    refusal_recall: float
    sufficient_rate: float  # fraction of answerable queries deemed sufficient at this threshold


def _sufficient_at(
    confidence: float,
    strong: int,
    min_confidence: float,
    strong_confidence: float,
    min_candidates: int,
) -> bool:
    """Mirrors HybridRetriever's gate (engine.py): same three-branch logic, given pre-computed inputs."""
    if confidence < min_confidence:
        return False
    if confidence < strong_confidence and strong < min_candidates:
        return False
    return True


def sweep_confidence_thresholds(
    per_query: list[dict[str, Any]],
    *,
    grid: list[float] | None = None,
    strong_confidence: float = 0.6,
    min_candidates: int = 2,
) -> list[ThresholdPoint]:
    """``per_query`` rows need ``confidence``, ``strong_count`` (candidates scoring >= 0.20), and
    ``expected_chunk_ids`` (empty list = distractor / should refuse).
    """
    grid = grid or [round(x * 0.05, 2) for x in range(1, 19)]  # 0.05 .. 0.90
    points: list[ThresholdPoint] = []
    for threshold in grid:
        counts = RefusalCounts()
        sufficient_answerable = 0
        n_answerable = 0
        for row in per_query:
            confidence = float(row.get("confidence", 0.0))
            strong = int(row.get("strong_count", 1 if confidence >= 0.20 else 0))
            expected_answerable = bool(row.get("expected_chunk_ids"))
            sufficient = _sufficient_at(
                confidence, strong, threshold, strong_confidence, min_candidates
            )
            counts = counts.add(expected_answerable=expected_answerable, did_answer=sufficient)
            if expected_answerable:
                n_answerable += 1
                sufficient_answerable += int(sufficient)
        points.append(
            ThresholdPoint(
                min_confidence=threshold,
                refusal_precision=counts.precision,
                refusal_recall=counts.recall,
                sufficient_rate=(sufficient_answerable / n_answerable) if n_answerable else 0.0,
            )
        )
    return points


def recommend_threshold(
    points: list[ThresholdPoint], *, min_sufficient_rate: float = 0.8
) -> float | None:
    """Highest ``min_confidence`` that still keeps ``sufficient_rate`` for answerable queries above
    ``min_sufficient_rate`` — i.e. the strictest refusal gate that doesn't start refusing real answers.
    """
    candidates = [p for p in points if p.sufficient_rate >= min_sufficient_rate]
    if not candidates:
        return None
    return max(p.min_confidence for p in candidates)
