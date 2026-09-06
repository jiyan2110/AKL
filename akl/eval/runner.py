"""Eval runner (PRD §12.1, §12.4): run every QA pair through real retrieval, score, aggregate.

Reuses the exact ``RAGService`` used in production — the eval harness measures the system as
deployed, not a separate code path. ``run_eval`` is synchronous and single-process; it is meant
for corpora in the thousands-of-chunks range (CI/staging), not a full production benchmark (that
is ``akl.eval.load`` / Locust, a different tool for a different question).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from akl.eval.metrics import RetrievalScore, aggregate, faithfulness_score, score_retrieval
from akl.rag.service import RAGService
from akl.security.principal import Principal


@dataclass
class EvalReport:
    version: str
    k: int
    aggregate: dict[str, float] = field(default_factory=dict)
    per_query: list[dict[str, Any]] = field(default_factory=list)
    faithfulness_mean: float | None = None
    duration_s: float = 0.0
    thresholds: dict[str, float] = field(default_factory=dict)
    passed: bool = True
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "k": self.k,
            "aggregate": self.aggregate,
            "faithfulness_mean": self.faithfulness_mean,
            "duration_s": round(self.duration_s, 2),
            "thresholds": self.thresholds,
            "passed": self.passed,
            "failures": self.failures,
            "per_query": self.per_query,
        }


def run_eval(
    rag: RAGService,
    qa_pairs: list[dict[str, Any]],
    *,
    principal: Principal | None = None,
    k: int = 10,
    mode: str = "hybrid",
    check_answers: bool = False,
    thresholds: dict[str, float] | None = None,
) -> EvalReport:
    """Run every QA pair through ``rag.search`` (and optionally ``rag.answer``), score, aggregate.

    ``thresholds`` — e.g. ``{"recall_at_10": 0.85, "mrr": 0.70, "refusal_precision": 0.8}`` — are
    checked against the aggregate at the end; any miss is recorded in ``failures`` and flips
    ``passed`` to False (used as the CI/nightly gate).
    """
    principal = principal or Principal.dev()
    start = time.perf_counter()
    scores: list[RetrievalScore] = []
    per_query: list[dict[str, Any]] = []
    faithfulness: list[float] = []
    version = str(qa_pairs[0]["version"]) if qa_pairs else "unknown"

    for qa in qa_pairs:
        question = str(qa["question"])
        expected_chunks = [str(c) for c in (qa.get("expected_chunk_ids") or [])]
        result = rag.search(question, principal, mode=mode, k=k, rerank=True)
        retrieved = [str(item["chunk_id"]) for item in result.results]
        score = score_retrieval(str(qa["qa_id"]), retrieved, expected_chunks, k=k)
        scores.append(score)
        row: dict[str, Any] = {
            "qa_id": score.qa_id,
            "question": question,
            "expected_chunk_ids": expected_chunks,
            "retrieved_chunk_ids": retrieved[:k],
            "recall_at_k": score.recall_at_k,
            "mrr": score.mrr,
            "ndcg_at_k": score.ndcg_at_k,
            "confidence": result.retrieval.confidence,
            "sufficient": result.retrieval.sufficient,
        }
        if check_answers:
            answer = rag.answer(question, principal, request_id=None, persist_trace=False)
            # AnswerResponse doesn't carry the raw uncited-ratio float, only the flags the citation
            # engine already derived from it (low_faithfulness/unsupported_token/no_citations); use
            # citation presence as the base signal and let those flags apply the real penalty.
            base_ratio = 0.0 if answer.citations else 1.0
            faith = faithfulness_score(base_ratio, answer.flags)
            faithfulness.append(faith)
            row["mode"] = answer.mode
            row["faithfulness"] = faith
            row["answer_present"] = answer.answer is not None
        per_query.append(row)

    agg = aggregate(scores, k=k)
    report = EvalReport(
        version=version,
        k=k,
        aggregate=agg,
        per_query=per_query,
        faithfulness_mean=(sum(faithfulness) / len(faithfulness)) if faithfulness else None,
        duration_s=time.perf_counter() - start,
        thresholds=thresholds or {},
    )
    failures = []
    for metric, minimum in (thresholds or {}).items():
        actual = agg.get(metric, report.faithfulness_mean if metric == "faithfulness" else None)
        if actual is None or actual < minimum:
            failures.append(f"{metric}={actual} < {minimum}")
    report.failures = failures
    report.passed = not failures
    return report
