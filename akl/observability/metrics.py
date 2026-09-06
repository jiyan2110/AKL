"""Prometheus metric catalog (PRD Chapter 8, Appendix F).

Two registries:
* ``API_REGISTRY`` — long-lived, scraped at ``GET /metrics`` (unchanged from Batch E; the names
  and labels already shipped are preserved so existing dashboards/alerts don't break).
* ``PIPELINE_REGISTRY`` — created fresh per Airflow task / CLI pipeline invocation and pushed to
  the Pushgateway with ``push_to_gateway`` (grouping key = job + dag_id + task_id), because batch
  jobs are not scraped — they push once and exit (Prometheus Pushgateway pattern).
"""

from __future__ import annotations

import logging
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.exposition import pushadd_to_gateway

log = logging.getLogger("akl.observability.metrics")

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
DURATION_BUCKETS_LONG = (1, 5, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600, 7200)

# ---------------------------------------------------------------------------
# API registry (scraped) — unchanged names from Batch E
# ---------------------------------------------------------------------------
REGISTRY = CollectorRegistry()

HTTP_REQUESTS = Counter(
    "akl_http_requests_total", "HTTP requests", ["method", "route", "status"], registry=REGISTRY
)
HTTP_LATENCY = Histogram(
    "akl_http_request_duration_seconds",
    "HTTP latency",
    ["method", "route"],
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)
HTTP_INFLIGHT = Gauge(
    "akl_http_requests_inflight", "In-flight requests", ["route"], registry=REGISTRY
)
RATE_LIMITED = Counter(
    "akl_http_rate_limited_total", "429 responses", ["route_class"], registry=REGISTRY
)
AUTH_FAILURES = Counter(
    "akl_auth_failures_total", "401/403 responses", ["reason"], registry=REGISTRY
)
SEARCH_REQUESTS = Counter(
    "akl_search_requests_total", "Search requests", ["mode", "intent"], registry=REGISTRY
)
SEARCH_LATENCY = Histogram(
    "akl_search_latency_seconds",
    "Search latency by stage",
    ["stage"],
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)
RERANK_CONFIDENCE = Histogram(
    "akl_rerank_confidence",
    "Top rerank score",
    ["intent"],
    buckets=tuple(i / 10 for i in range(11)),
    registry=REGISTRY,
)
INSUFFICIENT = Counter("akl_insufficient_evidence_total", "Refusals", ["reason"], registry=REGISTRY)
CHAT_REQUESTS = Counter(
    "akl_chat_requests_total", "Chat requests", ["mode", "stream"], registry=REGISTRY
)
LLM_LATENCY = Histogram(
    "akl_llm_latency_seconds", "LLM latency", ["phase"], buckets=LATENCY_BUCKETS, registry=REGISTRY
)
ANSWER_FLAGS = Counter("akl_answer_flags_total", "Answer flags", ["flag"], registry=REGISTRY)
ANSWER_CITATIONS = Histogram(
    "akl_answer_citations",
    "Citations per answer",
    buckets=(0, 1, 2, 3, 4, 5, 6, 8, 10),
    registry=REGISTRY,
)
SSE_ACTIVE = Gauge("akl_sse_streams_active", "Active SSE streams", registry=REGISTRY)
UPLOADS = Counter("akl_uploads_total", "Uploads", ["source_type", "outcome"], registry=REGISTRY)
UPLOAD_BYTES = Counter(
    "akl_upload_bytes_total", "Uploaded bytes", ["source_type"], registry=REGISTRY
)

# New in Batch G — retrieval backend health and freshness, scraped alongside the rest.
RETRIEVAL_DEGRADED = Counter(
    "akl_retrieval_degraded_total",
    "Dense/sparse backend failures seen by the API",
    ["backend"],
    registry=REGISTRY,
)
DAG_LAST_SUCCESS_TIMESTAMP = Gauge(
    "akl_dag_last_success_timestamp_seconds",
    "Unix time of the last successful run per DAG",
    ["dag_id"],
    registry=REGISTRY,
)
DAG_STALE = Gauge(
    "akl_dag_stale",
    "1 if the DAG's last success is older than its configured staleness threshold",
    ["dag_id"],
    registry=REGISTRY,
)
LINEAGE_EDGES_WRITTEN = Counter(
    "akl_lineage_edges_written_total", "Lineage edges recorded", ["edge_type"], registry=REGISTRY
)


def render() -> bytes:
    return generate_latest(REGISTRY)


def observe_search(
    mode: str,
    intent: str,
    timings_ms: dict[str, float],
    confidence: float,
    sufficient: bool,
    reason: str | None,
) -> None:
    SEARCH_REQUESTS.labels(mode=mode, intent=intent).inc()
    for stage, ms in timings_ms.items():
        SEARCH_LATENCY.labels(stage=stage).observe(ms / 1000.0)
    RERANK_CONFIDENCE.labels(intent=intent).observe(max(0.0, min(1.0, confidence)))
    if not sufficient:
        INSUFFICIENT.labels(reason=reason or "unknown").inc()


# ---------------------------------------------------------------------------
# Pipeline registry (pushed) — one instance per task_scope(), pushed once at task end
# ---------------------------------------------------------------------------
class PipelineMetrics:
    """Built fresh inside ``task_scope``; ``push()`` sends it to the Pushgateway (best-effort)."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.task_duration = Histogram(
            "akl_task_duration_seconds",
            "Airflow/CLI task duration",
            ["dag_id", "task_id", "state"],
            buckets=DURATION_BUCKETS_LONG,
            registry=self.registry,
        )
        self.task_rows_in = Gauge(
            "akl_task_rows_in",
            "Rows considered by the task",
            ["dag_id", "task_id"],
            registry=self.registry,
        )
        self.task_rows_out = Gauge(
            "akl_task_rows_out",
            "Rows produced by the task",
            ["dag_id", "task_id"],
            registry=self.registry,
        )
        self.ingestion_fetched = Counter(
            "akl_ingestion_fetched_total",
            "Documents fetched",
            ["connector_id"],
            registry=self.registry,
        )
        self.ingestion_quarantined = Counter(
            "akl_ingestion_quarantined_total", "Documents quarantined", registry=self.registry
        )
        self.ingestion_quarantine_ratio = Gauge(
            "akl_ingestion_quarantine_ratio", "Quarantine ratio for the run", registry=self.registry
        )
        self.chunking_documents = Gauge(
            "akl_chunking_documents",
            "Documents chunked/unchanged/failed",
            ["status"],
            registry=self.registry,
        )
        self.chunking_ops = Counter(
            "akl_chunking_ops_total", "Chunk diff outcomes", ["kind"], registry=self.registry
        )
        self.embedding_backlog = Gauge(
            "akl_embedding_backlog",
            "Chunks lacking a current-version embedding",
            registry=self.registry,
        )
        self.embedding_coverage = Gauge(
            "akl_embedding_coverage",
            "Fraction of active units with a current embedding",
            registry=self.registry,
        )
        self.embedding_cache_hit_ratio = Gauge(
            "akl_embedding_cache_hit_ratio",
            "Cache hits / backlog for the run",
            registry=self.registry,
        )
        self.embedding_throughput = Gauge(
            "akl_embedding_throughput_chunks_per_second",
            "Chunks embedded per second",
            registry=self.registry,
        )
        self.qdrant_drift = Gauge(
            "akl_qdrant_drift",
            "Qdrant point count minus Gold active unit count after sync",
            registry=self.registry,
        )
        self.qdrant_points = Gauge(
            "akl_qdrant_points", "Qdrant collection point count", registry=self.registry
        )
        self.bm25_documents = Gauge(
            "akl_bm25_documents", "Documents in the current BM25 index", registry=self.registry
        )
        self.gate_failures = Counter(
            "akl_gate_failures_total",
            "Quality gate failures",
            ["dag_id", "gate"],
            registry=self.registry,
        )
        self.maintenance_bytes_reclaimed = Counter(
            "akl_maintenance_bytes_reclaimed_total",
            "Bytes freed by compaction/retention",
            ["operation"],
            registry=self.registry,
        )

    def push(self, url: str, *, job: str, dag_id: str, task_id: str) -> None:
        if not url:
            return
        try:
            pushadd_to_gateway(
                url,
                job=job,
                registry=self.registry,
                grouping_key={"dag_id": dag_id, "task_id": task_id},
            )
        except Exception as exc:  # pragma: no cover - pushgateway is optional infra
            log.warning("pushgateway_unreachable", extra={"url": url, "error": str(exc)})


def apply_task_metrics(
    pm: PipelineMetrics, *, dag_id: str, task_id: str, out: dict[str, Any]
) -> None:
    """Fill in the metrics a task's result dict implies (best-effort; unknown keys are ignored)."""
    if "rows_in" in out and out["rows_in"] is not None:
        pm.task_rows_in.labels(dag_id=dag_id, task_id=task_id).set(float(out["rows_in"]))
    if "rows_out" in out and out["rows_out"] is not None:
        pm.task_rows_out.labels(dag_id=dag_id, task_id=task_id).set(float(out["rows_out"]))
    if "fetched" in out and "connector_id" in out:
        pm.ingestion_fetched.labels(connector_id=out["connector_id"]).inc(float(out["fetched"]))
    if "quarantined" in out and "quarantine_ratio" in out:
        pm.ingestion_quarantined.inc(float(out["quarantined"]))
        pm.ingestion_quarantine_ratio.set(float(out["quarantine_ratio"]))
    if "documents_chunked" in out:
        pm.chunking_documents.labels(status="chunked").set(float(out["documents_chunked"]))
        pm.chunking_documents.labels(status="unchanged").set(
            float(out.get("documents_unchanged", 0))
        )
        pm.chunking_documents.labels(status="failed").set(float(out.get("documents_failed", 0)))
        for kind in ("unchanged", "modified", "moved", "added", "removed"):
            if kind in out:
                pm.chunking_ops.labels(kind=kind).inc(float(out[kind]))
    if "coverage" in out and "backlog" in out:
        pm.embedding_coverage.set(float(out["coverage"]))
        pm.embedding_backlog.set(float(out.get("remaining_backlog", out["backlog"])))
        if out["backlog"]:
            pm.embedding_cache_hit_ratio.set(
                float(out.get("cache_hits", 0)) / float(out["backlog"])
            )
        if "throughput_cps" in out:
            pm.embedding_throughput.set(float(out["throughput_cps"]))
    if "drift" in out:
        pm.qdrant_drift.set(float(out["drift"]))
        if "after" in out:
            pm.qdrant_points.set(float(out["after"]))
    if "documents" in out and "terms" in out:
        pm.bm25_documents.set(float(out["documents"]))
    if "bytes_before" in out and "bytes_after" in out:
        pm.maintenance_bytes_reclaimed.labels(operation=task_id).inc(
            max(0.0, float(out["bytes_before"]) - float(out["bytes_after"]))
        )
