"""Sources and jobs API (PRD §10.4, §10.8 job status)."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Request

from akl.api.deps import AppState, get_request_id, get_state, rate_limited, scoped
from akl.api.schemas.documents import GitHubSyncRequest, JobStatus, SourceStatus, TriggerResponse
from akl.db.repositories.connector_state import ConnectorStateRepository
from akl.errors import AKLError
from akl.ingestion.service import IngestionService
from akl.lakehouse.engine import DuckDBEngine
from akl.pipelines.local import run_post_ingest
from akl.security.principal import Principal

router = APIRouter(prefix="/v1", tags=["sources"])


class SourceNotFoundError(AKLError):
    code = "AKL-E3041"
    http_status = 404
    retryable = False


class JobNotFoundError(AKLError):
    code = "AKL-E7002"
    http_status = 404
    retryable = False


def _run_connector_job(state: AppState, run_id: str, connector_id: str) -> None:
    state.jobs.update(run_id, state="running", stages=["fetch"])
    if state.db is None:
        state.jobs.update(
            run_id, state="failed", errors=[{"code": "AKL-E3023", "error": "no database"}]
        )
        return
    try:
        with state.lock, DuckDBEngine(state.settings) as engine:
            ingest = IngestionService(state.settings, engine, state.db)
            fetch = ingest.run_connector(connector_id, run_id=run_id)
        report = run_post_ingest(state.settings, state.db, run_id=run_id)
        summary: dict[str, Any] = {
            "fetched": fetch.fetched,
            "deduplicated": fetch.deduplicated,
            "failed": fetch.failed,
            "deletions": len(fetch.deletions),
            **report.as_dict(),
        }
        state.jobs.update(
            run_id,
            state="failed" if report.errors or fetch.failed else "succeeded",
            stages=["fetch", *report.stages],
            report=summary,
            errors=[*fetch.failures, *report.errors],
        )
        reload = getattr(state.rag, "reload_indexes", None)
        if callable(reload) and not report.errors:
            with state.lock:
                reload()
    except (AKLError, KeyError) as exc:
        state.jobs.update(
            run_id,
            state="failed",
            errors=[{"code": getattr(exc, "code", "AKL-E7001"), "error": str(exc)}],
        )


@router.get("/sources", response_model=list[SourceStatus])
def list_sources(
    state: AppState = Depends(get_state), principal: Principal = Depends(scoped("search:read"))
) -> list[SourceStatus]:
    if state.db is None or state.rag is None:
        return []
    out: list[SourceStatus] = []
    with state.lock, DuckDBEngine(state.settings) as engine:
        ingest = IngestionService(state.settings, engine, state.db)
        with state.db.session() as s:
            repo = ConnectorStateRepository(s)
            for cfg in ingest.connector_configs():
                health = ingest.connectors.create(cfg).health()
                row = repo.row(cfg.id)
                out.append(
                    SourceStatus(
                        id=cfg.id,
                        type=cfg.type,
                        enabled=cfg.enabled,
                        healthy=health.ok,
                        detail=health.detail,
                        documents_count=row.documents_count if row else 0,
                        last_run_id=row.last_run_id if row else None,
                        last_success_at=row.last_success_at if row else None,
                    )
                )
    return out


@router.post(
    "/sources/github/sync",
    response_model=TriggerResponse,
    status_code=202,
    dependencies=[Depends(rate_limited("default"))],
)
def github_sync(
    body: GitHubSyncRequest,
    request: Request,
    background: BackgroundTasks,
    state: AppState = Depends(get_state),
    principal: Principal = Depends(scoped("documents:write")),
    request_id: str = Depends(get_request_id),
) -> TriggerResponse:
    if state.db is None or state.rag is None:
        raise AKLError("service not ready")
    with state.lock, DuckDBEngine(state.settings) as engine:
        ingest = IngestionService(state.settings, engine, state.db)
        match = None
        for cfg in ingest.connector_configs():
            full = f"{getattr(cfg, 'owner', '')}/{getattr(cfg, 'repo', '')}"
            if (
                cfg.type == "github"
                and full.lower() == body.repo.lower()
                and (body.branch is None or getattr(cfg, "branch", None) == body.branch)
            ):
                match = cfg
    if match is None:
        raise SourceNotFoundError(
            f"no enabled github connector for {body.repo}", details={"repo": body.repo}
        )
    if match.owners and principal.subject not in match.owners and not principal.has_scope("*"):
        raise AKLError("principal is not an owner of this source", details={"owners": match.owners})
    run_id = f"api-{uuid.uuid4().hex[:12]}"
    state.jobs.create(run_id)
    background.add_task(_run_connector_job, state, run_id, match.id)
    return TriggerResponse(
        run_id=run_id,
        status_url=str(request.url_for("job_status", run_id=run_id)),
        message=f"sync of {body.repo} queued",
    )


@router.get("/jobs/{run_id}", response_model=JobStatus, name="job_status")
def job_status(
    run_id: str,
    state: AppState = Depends(get_state),
    principal: Principal = Depends(scoped("search:read")),
) -> JobStatus:
    job = state.jobs.get(run_id)
    if job is None:
        raise JobNotFoundError("job not found", details={"run_id": run_id})
    return JobStatus(
        run_id=job.run_id, state=job.state, stages=job.stages, report=job.report, errors=job.errors
    )
