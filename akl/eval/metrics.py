"""Retrieval and answer-quality metrics (PRD Chapter 12). All pure functions: given ranked ids and
a relevance set, or given answer/citation data, return a float or a small dataclass. No I/O here —
:mod:`akl.eval.runner` wires these to real retrieval/answer calls.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of ``relevant`` ids present anywhere in the first ``k`` of ``retrieved``."""
    if not relevant:
        return (
            1.0 if not retrieved[:k] else 0.0
        )  # nothing was relevant; retrieving nothing is "correct"
    hit = set(retrieved[:k]) & set(relevant)
    return len(hit) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    hit = set(top) & set(relevant)
    return len(hit) / len(top)


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """1/rank of the first relevant id in ``retrieved`` (0-based search), 0.0 if none found."""
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Binary-relevance nDCG@k (each relevant id worth gain 1, others 0)."""
    relevant_set = set(relevant)
    dcg = sum(
        1.0 / math.log2(i + 2) for i, doc_id in enumerate(retrieved[:k]) if doc_id in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else (1.0 if not relevant_set else 0.0)


@dataclass(frozen=True)
class RetrievalScore:
    qa_id: str
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    expected_empty: bool
    got_empty: bool

    @property
    def refusal_correct(self) -> bool | None:
        """For a distractor question (``expected_empty``), did retrieval correctly find nothing relevant?"""
        return self.got_empty if self.expected_empty else None


def score_retrieval(
    qa_id: str, retrieved: Sequence[str], relevant: Sequence[str], *, k: int
) -> RetrievalScore:
    expected_empty = not relevant
    return RetrievalScore(
        qa_id=qa_id,
        recall_at_k=recall_at_k(retrieved, relevant, k),
        precision_at_k=precision_at_k(retrieved, relevant, k),
        mrr=reciprocal_rank(retrieved, relevant),
        ndcg_at_k=ndcg_at_k(retrieved, relevant, k),
        expected_empty=expected_empty,
        got_empty=not retrieved,
    )


@dataclass(frozen=True)
class RefusalCounts:
    true_positive: int = 0  # correctly refused (no answer, none expected)
    false_positive: int = 0  # refused, but an answer was expected
    true_negative: int = 0  # answered, and an answer was expected
    false_negative: int = 0  # answered, but none was expected (hallucinated a source)

    @property
    def precision(self) -> float:
        """Of every refusal we issued, how many were correct (the question truly had no answer)?"""
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 1.0

    @property
    def recall(self) -> float:
        """Of every truly unanswerable question, how many did we correctly refuse?"""
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 1.0

    def add(self, *, expected_answerable: bool, did_answer: bool) -> RefusalCounts:
        if not expected_answerable and not did_answer:
            return RefusalCounts(
                self.true_positive + 1, self.false_positive, self.true_negative, self.false_negative
            )
        if expected_answerable and not did_answer:
            return RefusalCounts(
                self.true_positive, self.false_positive + 1, self.true_negative, self.false_negative
            )
        if expected_answerable and did_answer:
            return RefusalCounts(
                self.true_positive, self.false_positive, self.true_negative + 1, self.false_negative
            )
        return RefusalCounts(
            self.true_positive, self.false_positive, self.true_negative, self.false_negative + 1
        )


def aggregate(scores: Sequence[RetrievalScore], *, k: int) -> dict[str, float]:
    answerable = [s for s in scores if not s.expected_empty]
    refusal = RefusalCounts()
    for s in scores:
        refusal = refusal.add(expected_answerable=not s.expected_empty, did_answer=not s.got_empty)
    return {
        "n": len(scores),
        "n_answerable": len(answerable),
        "n_distractor": len(scores) - len(answerable),
        f"recall_at_{k}": _mean(s.recall_at_k for s in answerable),
        f"precision_at_{k}": _mean(s.precision_at_k for s in answerable),
        "mrr": _mean(s.mrr for s in answerable),
        f"ndcg_at_{k}": _mean(s.ndcg_at_k for s in answerable),
        "refusal_precision": refusal.precision,
        "refusal_recall": refusal.recall,
    }


def _mean(values: Iterable[float]) -> float:
    xs = list(values)
    return sum(xs) / len(xs) if xs else 0.0


def faithfulness_score(uncited_ratio: float, flags: Sequence[str]) -> float:
    """Cheap proxy for faithfulness (PRD §12.3): 1 - uncited_ratio, further penalised for known
    hallucination flags already computed by the answer pipeline (no separate NLI model needed —
    the citation/uncited-ratio machinery in akl.rag.citations already does the sentence-level work).
    """
    score = 1.0 - max(0.0, min(1.0, uncited_ratio))
    if "unsupported_token" in flags:
        score *= 0.5
    if "low_faithfulness" in flags:
        score *= 0.5
    return round(max(0.0, min(1.0, score)), 4)
