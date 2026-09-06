"""``/v1/admin/lineage`` — dataset-level run lineage and per-document trace (PRD §9.9)."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends

from akl.api.deps import AppState, get_state, scoped
from akl.db.repositories.lineage import LineageRepository
from akl.db.repositories.runs import RunRepository
from akl.errors import AKLError
from akl.security.principal import Principal

router = APIRouter(prefix="/v1/admin/lineage", tags=["admin"])


class LineageNotFoundError(AKLError):
    code = "AKL-E9010"
    http_status = 404
    retryable = False


def _edge_dict(edge: Any) -> dict[str, Any]:
    return {
        "run_id": edge.run_id,
        "task_id": edge.task_id,
        "input_dataset": edge.input_dataset,
        "input_partition": edge.input_partition,
        "output_dataset": edge.output_dataset,
        "output_partition": edge.output_partition,
        "rows_in": edge.rows_in,
        "rows_out": edge.rows_out,
        "created_at": edge.created_at.isoformat() if edge.created_at else None,
    }


@router.get("/runs/{run_id}")
def run_lineage(
    run_id: str,
    state: AppState = Depends(get_state),
    _: Principal = Depends(scoped("admin:reload")),
) -> dict[str, Any]:
    """Every dataset edge and task record for one pipeline run (Bronze→Silver→Gold chain)."""
    if state.db is None:
        raise LineageNotFoundError("no database configured")
    with state.db.session() as s:
        edges = [_edge_dict(e) for e in LineageRepository(s).for_run(run_id)]
        tasks = [
            {
                "task_id": t.task_id,
                "state": t.state,
                "rows_in": t.rows_in,
                "rows_out": t.rows_out,
                "metrics": t.metrics,
            }
            for t in RunRepository(s).tasks(run_id)
        ]
    if not edges and not tasks:
        raise LineageNotFoundError("run not found", details={"run_id": run_id})
    return {"run_id": run_id, "edges": edges, "tasks": tasks}


@router.get("/datasets/{dataset:path}")
def dataset_lineage(
    dataset: str,
    state: AppState = Depends(get_state),
    _: Principal = Depends(scoped("admin:reload")),
) -> dict[str, Any]:
    """Recent runs that produced a given dataset, e.g. ``silver/chunks`` or ``gold/chunk_embeddings``."""
    if state.db is None:
        return {"dataset": dataset, "edges": []}
    with state.db.session() as s:
        edges = [_edge_dict(e) for e in LineageRepository(s).for_dataset(dataset)]
    return {"dataset": dataset, "edges": edges}


@router.get("/documents/{document_id}")
def document_lineage(
    document_id: str,
    state: AppState = Depends(get_state),
    principal: Principal = Depends(scoped("search:read")),
) -> dict[str, Any]:
    """Live trace of one document: versions, current chunk count, embedding status breakdown.

    Subject to the same security-level/group visibility rule as ``GET /v1/documents/{id}``.
    """
    if state.db is None:
        raise LineageNotFoundError("no database configured")
    try:
        did = uuid.UUID(document_id)
    except ValueError as exc:
        raise LineageNotFoundError(
            "invalid document id", details={"document_id": document_id}
        ) from exc
    with state.db.session() as s:
        from akl.db.repositories.documents import DocumentRepository

        doc = DocumentRepository(s).get(did)
        if doc is None or not principal.can_read(
            doc.security_level, list(doc.allowed_groups or [])
        ):
            raise LineageNotFoundError("document not found", details={"document_id": document_id})
        trace = LineageRepository(s).document_trace(did)
    if trace is None:
        raise LineageNotFoundError("document not found", details={"document_id": document_id})
    return trace
