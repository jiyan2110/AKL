"""Search API (PRD §10.5)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from akl.api import metrics
from akl.api.deps import AppState, get_principal, get_request_id, get_state, rate_limited, scoped
from akl.api.schemas.search import (
    QueryInfo,
    Scores,
    SearchFilters,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from akl.rag.query.filters import MetadataFilters
from akl.security.principal import Principal

router = APIRouter(prefix="/v1", tags=["search"])


def to_filters(f: SearchFilters | None) -> MetadataFilters | None:
    if f is None:
        return None
    mf = MetadataFilters(
        source_types=list(f.source_type),
        repos=list(f.repo),
        chunk_types=list(f.chunk_type),
        code_languages=list(f.code_language),
        document_ids=list(f.document_id),
        updated_after=f.updated_after,
    )
    return None if mf.is_empty() else mf


@router.post(
    "/search", response_model=SearchResponse, dependencies=[Depends(rate_limited("search"))]
)
def search(
    body: SearchRequest,
    state: AppState = Depends(get_state),
    principal: Principal = Depends(scoped("search:read")),
    request_id: str = Depends(get_request_id),
) -> SearchResponse:
    with state.lock:
        res = state.rag.search(
            body.query,
            principal,
            mode=body.mode,
            k=body.k,
            filters=to_filters(body.filters),
            rerank=body.rerank,
            include_text=body.include_text,
            request_id=request_id,
        )
    r = res.retrieval
    metrics.observe_search(
        body.mode, res.query.intent.value, res.timings_ms, r.confidence, r.sufficient, r.reason
    )
    return SearchResponse(
        request_id=res.request_id,
        trace_id=res.trace_id,
        results=[
            SearchResult(**{**item, "scores": Scores(**item["scores"])}) for item in res.results
        ],
        query=QueryInfo(
            normalized=res.query.dense_text,
            corrected=res.query.sparse_text,
            intent=res.query.intent.value,
            entities=res.query.entities.as_dict(),
            filters_applied={
                "hard": res.query.hard_filters.as_dict(),
                "soft": res.query.soft_filters.as_dict(),
            },
        ),
        confidence=r.confidence,
        sufficient=r.sufficient,
        flags=[*getattr(state.rag, "flags", []), *r.flags],
        timings_ms=res.timings_ms,
        gold_snapshot_id=res.gold_snapshot_id,
    )


__all__ = ["router", "to_filters", "get_principal"]
