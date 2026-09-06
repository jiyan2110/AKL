"""Unit tests: eval metrics (recall/precision/MRR/nDCG, refusal precision/recall) — Milestone 53."""

from __future__ import annotations

import pytest

from akl.eval.metrics import (
    RefusalCounts,
    aggregate,
    faithfulness_score,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_retrieval,
)

pytestmark = pytest.mark.unit


def test_recall_precision_basic() -> None:
    retrieved = ["a", "b", "c", "d"]
    assert recall_at_k(retrieved, ["c"], 3) == 1.0
    assert recall_at_k(retrieved, ["c"], 2) == 0.0
    assert recall_at_k(retrieved, ["c", "z"], 4) == 0.5  # only one of two relevant ids present
    assert precision_at_k(retrieved, ["a", "b"], 2) == 1.0
    assert precision_at_k(retrieved, ["a"], 4) == 0.25
    assert precision_at_k([], ["a"], 3) == 0.0


def test_recall_on_empty_relevant_set_means_correct_refusal() -> None:
    # a distractor question: nothing is relevant, so retrieving nothing is the "correct" outcome
    assert recall_at_k([], [], 5) == 1.0
    assert recall_at_k(["x"], [], 5) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0
    assert reciprocal_rank(["a", "b", "c"], ["z"]) == 0.0
    assert reciprocal_rank([], ["z"]) == 0.0


def test_ndcg_ideal_and_non_ideal_ordering() -> None:
    assert ndcg_at_k(["c", "a", "b"], ["c"], 3) == 1.0  # relevant item is first: ideal
    assert ndcg_at_k(["a", "b", "c"], ["c"], 3) == pytest.approx(
        0.5, abs=1e-4
    )  # relevant item last of 3
    assert (
        ndcg_at_k(["a", "b"], [], 3) == 1.0
    )  # nothing relevant, nothing to rank -> trivially ideal


def test_score_retrieval_flags_distractor_correctly() -> None:
    hit = score_retrieval("q1", ["a", "b"], ["a"], k=2)
    assert hit.expected_empty is False
    assert hit.refusal_correct is None  # not a distractor question
    distractor_ok = score_retrieval("q2", [], [], k=2)
    assert distractor_ok.expected_empty is True
    assert distractor_ok.refusal_correct is True
    distractor_missed = score_retrieval("q3", ["x"], [], k=2)
    assert distractor_missed.refusal_correct is False


def test_refusal_counts_precision_and_recall_are_not_swapped() -> None:
    """Regression: precision and recall were accidentally swapped once during development."""
    counts = RefusalCounts()
    counts = counts.add(expected_answerable=False, did_answer=False)  # TP: correct refusal
    counts = counts.add(expected_answerable=False, did_answer=False)  # TP: correct refusal
    counts = counts.add(
        expected_answerable=False, did_answer=True
    )  # FN: missed refusal (hallucinated)
    counts = counts.add(
        expected_answerable=True, did_answer=False
    )  # FP: wrongly refused an answerable one
    counts = counts.add(expected_answerable=True, did_answer=True)  # TN: correctly answered
    assert (
        counts.true_positive,
        counts.false_positive,
        counts.true_negative,
        counts.false_negative,
    ) == (2, 1, 1, 1)
    # precision = of refusals issued (TP+FP=3), how many were correct (TP=2) -> 2/3
    assert counts.precision == pytest.approx(2 / 3)
    # recall = of truly-unanswerable cases (TP+FN=3), how many were caught (TP=2) -> 2/3
    assert counts.recall == pytest.approx(2 / 3)


def test_refusal_counts_edge_cases_default_to_one() -> None:
    empty = RefusalCounts()
    assert empty.precision == 1.0
    assert empty.recall == 1.0


def test_aggregate_separates_answerable_from_distractor_and_computes_refusal_stats() -> None:
    s1 = score_retrieval("q1", ["a", "b", "c"], ["c"], k=3)  # answerable, found
    s2 = score_retrieval("q2", [], [], k=3)  # distractor, correctly refused
    s3 = score_retrieval("q3", ["x"], [], k=3)  # distractor, wrongly answered
    agg = aggregate([s1, s2, s3], k=3)
    assert agg["n"] == 3
    assert agg["n_answerable"] == 1
    assert agg["n_distractor"] == 2
    assert agg["recall_at_3"] == 1.0  # averaged only over the one answerable query
    assert agg["refusal_precision"] == 1.0  # the only refusal we issued (q2) was correct
    assert agg["refusal_recall"] == 0.5  # caught 1 of 2 truly-unanswerable cases


def test_aggregate_handles_no_scores() -> None:
    agg = aggregate([], k=5)
    assert agg["n"] == 0
    assert agg["recall_at_5"] == 0.0
    assert agg["refusal_precision"] == 1.0


def test_faithfulness_score_penalises_known_flags() -> None:
    assert faithfulness_score(0.0, []) == 1.0
    assert faithfulness_score(0.5, []) == 0.5
    assert faithfulness_score(0.0, ["unsupported_token"]) == 0.5
    assert faithfulness_score(0.0, ["low_faithfulness"]) == 0.5
    assert faithfulness_score(0.0, ["unsupported_token", "low_faithfulness"]) == 0.25
    assert faithfulness_score(1.0, []) == 0.0
    assert faithfulness_score(2.0, []) == 0.0  # clamped
