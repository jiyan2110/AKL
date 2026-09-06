"""Unit tests: benchmark harness percentile math and threshold gating (Milestone 55)."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest

from akl.eval.benchmark import LatencyStats, run_benchmark
from akl.security.principal import Principal

pytestmark = pytest.mark.unit


def test_latency_stats_percentiles_match_hand_computation() -> None:
    stats = LatencyStats.from_samples([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    assert stats.p50_ms == pytest.approx(55.0)
    assert stats.p95_ms == pytest.approx(95.5)
    assert stats.min_ms == 10
    assert stats.max_ms == 100
    assert stats.mean_ms == pytest.approx(55.0)


def test_latency_stats_single_sample_and_empty() -> None:
    single = LatencyStats.from_samples([42.0])
    assert (single.p50_ms, single.p95_ms, single.p99_ms) == (42.0, 42.0, 42.0)
    empty = LatencyStats.from_samples([])
    assert empty.n == 0
    assert empty.p50_ms == 0.0


class FakeBenchRAG:
    def __init__(self, *, search_delay_s: float = 0.0) -> None:
        self.provider = SimpleNamespace(embedding_version="fake-v1")
        self.reranker = SimpleNamespace(name="lexical")
        self.search_delay_s = search_delay_s
        self.search_calls = 0
        self.answer_calls = 0

    def search(self, query: str, principal: Principal, **kw: Any) -> Any:
        self.search_calls += 1
        if self.search_delay_s:
            time.sleep(self.search_delay_s)
        return SimpleNamespace(timings_ms={"dense": 1.0, "sparse": 0.5})

    def answer(self, query: str, principal: Principal, **kw: Any) -> Any:
        self.answer_calls += 1
        return SimpleNamespace(answer="ok")


def test_run_benchmark_warms_up_and_counts_calls() -> None:
    rag = FakeBenchRAG()
    report = run_benchmark(rag, queries=("q1", "q2"), repeats=3)
    # warm-up (1 call) + repeats * len(queries) = 1 + 6 = 7
    assert rag.search_calls == 7
    assert report.search.n == 6  # warm-up call itself is excluded from the timed samples
    assert report.embedding_version == "fake-v1"
    assert report.reranker == "lexical"
    assert "dense" in report.per_stage
    assert rag.answer_calls == 0


def test_run_benchmark_include_answer() -> None:
    rag = FakeBenchRAG()
    report = run_benchmark(rag, queries=("q1",), repeats=2, include_answer=True)
    assert rag.answer_calls == 2
    assert report.answer is not None
    assert report.answer.n == 2


def test_run_benchmark_thresholds_pass_and_fail() -> None:
    rag = FakeBenchRAG()
    ok = run_benchmark(rag, queries=("q1",), repeats=2, thresholds={"p95_ms": 10_000})
    assert ok.passed is True
    bad = run_benchmark(rag, queries=("q1",), repeats=2, thresholds={"p95_ms": 0.0})
    assert bad.passed is False
    assert "p95_ms" in bad.failures[0]


def test_run_benchmark_answer_prefixed_threshold() -> None:
    rag = FakeBenchRAG()
    report = run_benchmark(
        rag, queries=("q1",), repeats=1, include_answer=True, thresholds={"answer_p95_ms": 0.0}
    )
    assert report.passed is False
    assert report.failures[0].startswith("answer_p95_ms")


def test_report_markdown_contains_key_sections() -> None:
    rag = FakeBenchRAG()
    report = run_benchmark(
        rag, queries=("q1",), repeats=1, include_answer=True, thresholds={"p95_ms": 10_000}
    )
    md = report.to_markdown()
    assert "# AKL Benchmark Report" in md
    assert "## Search latency (ms)" in md
    assert "## Answer (chat) latency (ms)" in md
    assert "## Per-stage p95 (ms)" in md
    assert "fake-v1" in md
    as_dict = report.as_dict()
    assert as_dict["search"]["n"] == 1
