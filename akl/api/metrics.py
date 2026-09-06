"""API metrics — re-exports the shared catalog (PRD §8.3, Appendix F.1–F.3).

The metric objects live in :mod:`akl.observability.metrics` (one process-wide registry, shared
with pipeline-side helpers that need the same names for documentation purposes); this module is
kept so every existing ``from akl.api import metrics`` call site is unaffected.
"""

from __future__ import annotations

from akl.observability.metrics import (
    ANSWER_CITATIONS,
    ANSWER_FLAGS,
    AUTH_FAILURES,
    CHAT_REQUESTS,
    DAG_LAST_SUCCESS_TIMESTAMP,
    DAG_STALE,
    HTTP_INFLIGHT,
    HTTP_LATENCY,
    HTTP_REQUESTS,
    INSUFFICIENT,
    LINEAGE_EDGES_WRITTEN,
    LLM_LATENCY,
    RATE_LIMITED,
    REGISTRY,
    RERANK_CONFIDENCE,
    RETRIEVAL_DEGRADED,
    SEARCH_LATENCY,
    SEARCH_REQUESTS,
    SSE_ACTIVE,
    UPLOAD_BYTES,
    UPLOADS,
    observe_search,
    render,
)

__all__ = [
    "ANSWER_CITATIONS",
    "ANSWER_FLAGS",
    "AUTH_FAILURES",
    "CHAT_REQUESTS",
    "DAG_LAST_SUCCESS_TIMESTAMP",
    "DAG_STALE",
    "HTTP_INFLIGHT",
    "HTTP_LATENCY",
    "HTTP_REQUESTS",
    "INSUFFICIENT",
    "LINEAGE_EDGES_WRITTEN",
    "LLM_LATENCY",
    "RATE_LIMITED",
    "REGISTRY",
    "RERANK_CONFIDENCE",
    "RETRIEVAL_DEGRADED",
    "SEARCH_LATENCY",
    "SEARCH_REQUESTS",
    "SSE_ACTIVE",
    "UPLOAD_BYTES",
    "UPLOADS",
    "observe_search",
    "render",
]
