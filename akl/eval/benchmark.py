"""Benchmark harness (PRD §11.7, NFR-01): repeated in-process search/answer calls, latency
percentiles, written as a markdown report under ``docs/benchmarks/``.

This measures the same ``RAGService`` the API uses, in-process — no HTTP overhead — so it isolates
retrieval/generation cost from network/ASGI cost (that's what the Locust load test in
``tests/load/`` is for). Use this to catch a regression in retrieval latency itself (e.g. a slower
reranker, a missing Qdrant index) between releases.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from akl.rag.service import RAGService
from akl.security.principal import Principal

DEFAULT_QUERIES: tuple[str, ...] = (
    "how are nightly postgres backups taken",
    "what does make seed do",
    "how is the qdrant collection synced",
    "what happens during a hard delete",
    "how does the embedding cache work",
    "what is the chunk_config_hash",
    "how do I mint a development JWT",
    "what does the ingestion quality gate check",
)


@dataclass
class LatencyStats:
    n: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float

    @classmethod
    def from_samples(cls, samples_ms: list[float]) -> LatencyStats:
        if not samples_ms:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        ordered = sorted(samples_ms)
        return cls(
            n=len(ordered),
            p50_ms=_percentile(ordered, 0.50),
            p95_ms=_percentile(ordered, 0.95),
            p99_ms=_percentile(ordered, 0.99),
            mean_ms=statistics.fmean(ordered),
            min_ms=ordered[0],
            max_ms=ordered[-1],
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "p50_ms": round(self.p50_ms, 1),
            "p95_ms": round(self.p95_ms, 1),
            "p99_ms": round(self.p99_ms, 1),
            "mean_ms": round(self.mean_ms, 1),
            "min_ms": round(self.min_ms, 1),
            "max_ms": round(self.max_ms, 1),
        }


def _percentile(ordered: list[float], p: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    rank = p * (len(ordered) - 1)
    lo, hi = int(rank), min(int(rank) + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


@dataclass
class BenchmarkReport:
    generated_at: str
    embedding_version: str
    reranker: str | None
    search: LatencyStats
    answer: LatencyStats | None
    per_stage: dict[str, LatencyStats] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    passed: bool = True
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "embedding_version": self.embedding_version,
            "reranker": self.reranker,
            "search": self.search.as_dict(),
            "answer": self.answer.as_dict() if self.answer else None,
            "per_stage": {k: v.as_dict() for k, v in self.per_stage.items()},
            "thresholds": self.thresholds,
            "passed": self.passed,
            "failures": self.failures,
        }

    def to_markdown(self) -> str:
        lines = [
            "# AKL Benchmark Report",
            "",
            f"- Generated: {self.generated_at}",
            f"- Embedding version: `{self.embedding_version}`",
            f"- Reranker: `{self.reranker}`",
            f"- Status: {'✅ PASS' if self.passed else '❌ FAIL'}",
            "",
            "## Search latency (ms)",
            "",
            "| n | p50 | p95 | p99 | mean | min | max |",
            "|---|---|---|---|---|---|---|",
            f"| {self.search.n} | {self.search.p50_ms:.1f} | {self.search.p95_ms:.1f} | {self.search.p99_ms:.1f} | {self.search.mean_ms:.1f} | {self.search.min_ms:.1f} | {self.search.max_ms:.1f} |",
        ]
        if self.answer:
            lines += [
                "",
                "## Answer (chat) latency (ms)",
                "",
                "| n | p50 | p95 | p99 | mean | min | max |",
                "|---|---|---|---|---|---|---|",
                f"| {self.answer.n} | {self.answer.p50_ms:.1f} | {self.answer.p95_ms:.1f} | {self.answer.p99_ms:.1f} | {self.answer.mean_ms:.1f} | {self.answer.min_ms:.1f} | {self.answer.max_ms:.1f} |",
            ]
        if self.per_stage:
            lines += [
                "",
                "## Per-stage p95 (ms)",
                "",
                "| stage | p50 | p95 | p99 |",
                "|---|---|---|---|",
            ]
            lines += [
                f"| {name} | {s.p50_ms:.1f} | {s.p95_ms:.1f} | {s.p99_ms:.1f} |"
                for name, s in self.per_stage.items()
            ]
        if self.thresholds:
            lines += ["", "## Thresholds", ""]
            lines += [f"- `{k}`: {v}" for k, v in self.thresholds.items()]
            if self.failures:
                lines += ["", "**Failures:**"] + [f"- {f}" for f in self.failures]
        return "\n".join(lines) + "\n"


def run_benchmark(
    rag: RAGService,
    *,
    queries: tuple[str, ...] = DEFAULT_QUERIES,
    repeats: int = 5,
    principal: Principal | None = None,
    include_answer: bool = False,
    thresholds: dict[str, float] | None = None,
) -> BenchmarkReport:
    """Runs ``search()`` (and optionally ``answer()``) ``repeats`` times per query, warms up once."""
    principal = principal or Principal.dev()
    if queries:
        rag.search(
            queries[0], principal, k=8
        )  # warm-up: exclude model/connection cold-start from the samples

    search_ms: list[float] = []
    answer_ms: list[float] = []
    stage_ms: dict[str, list[float]] = {}
    for _ in range(repeats):
        for query in queries:
            t0 = time.perf_counter()
            result = rag.search(query, principal, k=8)
            search_ms.append((time.perf_counter() - t0) * 1000)
            for stage, ms in result.timings_ms.items():
                stage_ms.setdefault(stage, []).append(ms)
            if include_answer:
                t1 = time.perf_counter()
                rag.answer(query, principal, persist_trace=False)
                answer_ms.append((time.perf_counter() - t1) * 1000)

    search_stats = LatencyStats.from_samples(search_ms)
    answer_stats = LatencyStats.from_samples(answer_ms) if answer_ms else None
    per_stage = {name: LatencyStats.from_samples(samples) for name, samples in stage_ms.items()}

    report = BenchmarkReport(
        generated_at=datetime.now(UTC).isoformat(),
        embedding_version=rag.provider.embedding_version,
        reranker=getattr(rag.reranker, "name", None),
        search=search_stats,
        answer=answer_stats,
        per_stage=per_stage,
        thresholds=thresholds or {},
    )
    failures = []
    for metric, maximum in (thresholds or {}).items():
        target, field_name = (
            (report.answer, metric[len("answer_") :])
            if metric.startswith("answer_")
            else (report.search, metric)
        )
        if target is None:
            continue
        actual = getattr(target, field_name, None)
        if actual is not None and actual > maximum:
            failures.append(f"{metric}={actual:.1f}ms > {maximum}ms")
    report.failures = failures
    report.passed = not failures
    return report
