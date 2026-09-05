"""Prometheus metrics for the API (PRD §8.3, Appendix F.1–F.3). Full catalog arrives in Milestone 44."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

REGISTRY = CollectorRegistry()
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)

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
