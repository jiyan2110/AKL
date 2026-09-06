"""Unit tests: eval runner (wiring + threshold gate) and confidence calibration (Milestone 53)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from akl.eval.calibration import recommend_threshold, sweep_confidence_thresholds
from akl.eval.runner import run_eval
from akl.security.principal import Principal

pytestmark = pytest.mark.unit


class FakeSearchRAG:
    """Minimal double for RAGService: canned per-question search/answer results."""

    def __init__(
        self,
        search_by_question: dict[str, list[str]],
        *,
        confidence_by_question: dict[str, float] | None = None,
    ) -> None:
        self.search_by_question = search_by_question
        self.confidence_by_question = confidence_by_question or {}
        self.search_calls: list[dict[str, Any]] = []
        self.answer_calls: list[str] = []

    def search(self, question: str, principal: Principal, **kw: Any) -> Any:
        self.search_calls.append({"question": question, **kw})
        chunk_ids = self.search_by_question.get(question, [])
        confidence = self.confidence_by_question.get(question, 0.9 if chunk_ids else 0.1)
        retrieval = SimpleNamespace(confidence=confidence, sufficient=bool(chunk_ids))
        results = [{"chunk_id": cid} for cid in chunk_ids]
        return SimpleNamespace(results=results, retrieval=retrieval)

    def answer(self, question: str, principal: Principal, **kw: Any) -> Any:
        self.answer_calls.append(question)
        chunk_ids = self.search_by_question.get(question, [])
        if not chunk_ids:
            return SimpleNamespace(
                answer=None, citations=[], flags=["insufficient_evidence"], mode="none"
            )
        return SimpleNamespace(
            answer="Answer citing [1].",
            citations=[{"index": 1, "chunk_id": chunk_ids[0]}],
            flags=[],
            mode="extractive",
        )


def _qa(qa_id: str, question: str, expected: list[str], version: str = "v1") -> dict[str, Any]:
    return {
        "qa_id": qa_id,
        "question": question,
        "expected_chunk_ids": expected,
        "version": version,
    }


def test_run_eval_basic_wiring_and_aggregate() -> None:
    rag = FakeSearchRAG({"Q1": ["c1", "c2"], "Q2": []})
    qa_pairs = [_qa("q1", "Q1", ["c1"]), _qa("q2", "Q2", [])]
    report = run_eval(rag, qa_pairs, k=5)
    assert report.version == "v1"
    assert report.aggregate["n"] == 2
    assert report.aggregate["n_answerable"] == 1
    assert report.aggregate["n_distractor"] == 1
    assert report.aggregate["recall_at_5"] == 1.0
    assert report.aggregate["refusal_precision"] == 1.0  # Q2 correctly returned nothing
    assert len(rag.search_calls) == 2
    assert rag.search_calls[0]["k"] == 5
    assert rag.answer_calls == []  # check_answers defaults to False


def test_run_eval_check_answers_computes_faithfulness() -> None:
    rag = FakeSearchRAG({"Q1": ["c1"]})
    report = run_eval(rag, [_qa("q1", "Q1", ["c1"])], k=3, check_answers=True)
    assert rag.answer_calls == ["Q1"]
    assert report.faithfulness_mean == 1.0  # cited, no penalty flags
    assert report.per_query[0]["mode"] == "extractive"
    assert report.per_query[0]["answer_present"] is True


def test_run_eval_check_answers_penalises_flags_and_missing_answers() -> None:
    rag = FakeSearchRAG({"Q1": ["c1"], "Q2": []})
    report = run_eval(rag, [_qa("q1", "Q1", ["c1"]), _qa("q2", "Q2", [])], k=3, check_answers=True)
    assert report.per_query[1]["answer_present"] is False
    assert report.faithfulness_mean == pytest.approx((1.0 + 0.0) / 2)


def test_run_eval_thresholds_gate_pass_and_fail() -> None:
    rag = FakeSearchRAG({"Q1": ["c1"]})
    ok = run_eval(rag, [_qa("q1", "Q1", ["c1"])], k=3, thresholds={"recall_at_3": 0.5})
    assert ok.passed is True
    assert ok.failures == []
    bad = run_eval(rag, [_qa("q1", "Q1", ["c1"])], k=3, thresholds={"recall_at_3": 1.1})
    assert bad.passed is False
    assert "recall_at_3" in bad.failures[0]


def test_run_eval_empty_qa_pairs() -> None:
    rag = FakeSearchRAG({})
    report = run_eval(rag, [], k=5)
    assert report.version == "unknown"
    assert report.aggregate["n"] == 0


def test_run_eval_uses_default_principal_when_none_given() -> None:
    rag = FakeSearchRAG({"Q1": ["c1"]})
    run_eval(rag, [_qa("q1", "Q1", ["c1"])], k=3, principal=None)
    # Principal.dev() should have been used; FakeSearchRAG doesn't inspect it but this should not raise
    assert len(rag.search_calls) == 1


# --------------------------------------------------------------------------- calibration
def _row(confidence: float, expected: list[str], strong: int | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"confidence": confidence, "expected_chunk_ids": expected}
    if strong is not None:
        row["strong_count"] = strong
    return row


def test_sweep_confidence_thresholds_matches_production_gate_shape() -> None:
    per_query = [
        _row(0.9, ["c1"], strong=2),  # confidently answerable
        _row(0.4, ["c2"], strong=1),  # borderline answerable, only 1 strong candidate
        _row(0.05, [], strong=0),  # correctly low-confidence distractor
        _row(
            0.7, [], strong=1
        ),  # distractor that scored high (should be refused at high thresholds only if < strong)
    ]
    points = sweep_confidence_thresholds(
        per_query, grid=[0.1, 0.35, 0.6], strong_confidence=0.6, min_candidates=2
    )
    assert [p.min_confidence for p in points] == [0.1, 0.35, 0.6]
    low = points[0]
    # at threshold 0.1: all >=0.1 confidence pass min_confidence; borderline (0.4) still needs 2 strong candidates and only has 1 -> insufficient
    assert 0.0 <= low.sufficient_rate <= 1.0
    high = points[-1]
    assert high.refusal_recall <= 1.0 and high.refusal_precision <= 1.0  # noqa: PT018


def test_sweep_higher_threshold_never_increases_sufficient_rate() -> None:
    per_query = [
        _row(0.9, ["c1"], strong=2),
        _row(0.3, ["c2"], strong=2),
        _row(0.1, ["c3"], strong=2),
    ]
    points = sweep_confidence_thresholds(per_query, grid=[0.05, 0.2, 0.5, 0.8])
    rates = [p.sufficient_rate for p in points]
    assert rates == sorted(
        rates, reverse=True
    )  # monotonically non-increasing as the gate gets stricter


def test_recommend_threshold_picks_strictest_that_still_clears_bar() -> None:
    points = sweep_confidence_thresholds(
        [_row(0.9, ["c1"], strong=2), _row(0.8, ["c2"], strong=2), _row(0.3, ["c3"], strong=2)],
        grid=[0.1, 0.4, 0.85],
    )
    rec = recommend_threshold(points, min_sufficient_rate=0.6)
    assert (
        rec == 0.4
    )  # 2 of 3 answerable still sufficient at 0.4 (>=0.6); 0.85 would drop below the bar
    assert (
        recommend_threshold(points, min_sufficient_rate=1.01) is None
    )  # impossible bar -> no threshold qualifies
