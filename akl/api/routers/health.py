"""Health endpoints (PRD §10.7): liveness, readiness with dependency probes, dependencies detail."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Response

from akl.api.deps import AppState, get_state
from akl.api.schemas.health import DependencyStatus, Liveness, Readiness

router = APIRouter(prefix="/v1/health", tags=["health"])


def _probe(name: str, fn: object) -> DependencyStatus:
    start = time.perf_counter()
    try:
        detail = str(fn() or "")  # type: ignore[operator]
        return DependencyStatus(
            name=name,
            ok=True,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            detail=detail,
        )
    except Exception as exc:
        return DependencyStatus(
            name=name,
            ok=False,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            detail=f"{type(exc).__name__}: {exc}",
        )


def check_dependencies(state: AppState) -> list[DependencyStatus]:
    deps: list[DependencyStatus] = []
    if state.db is not None:
        db = state.db
        deps.append(_probe("postgres", lambda: db.ping().server_version))
    rag = state.rag
    if rag is not None:
        if getattr(rag, "qdrant", None) is not None:
            deps.append(
                _probe(
                    "qdrant",
                    lambda: (
                        f"alias→{rag.qdrant.schema.alias_target()} points={rag.qdrant.schema.status().points}"
                    ),
                )
            )
        else:
            deps.append(
                DependencyStatus(
                    name="qdrant", ok=False, latency_ms=0.0, detail="dense retrieval unavailable"
                )
            )
        deps.append(
            DependencyStatus(
                name="bm25",
                ok=getattr(rag, "bm25", None) is not None,
                latency_ms=0.0,
                detail=str(getattr(getattr(rag, "bm25", None), "version", "not loaded")),
            )
        )
        if getattr(rag, "io", None) is not None:
            deps.append(
                _probe(
                    "minio",
                    lambda: f"bucket={rag.io.bucket}" if rag.io.ensure_bucket() is None else "",
                )
            )
        deps.append(
            DependencyStatus(
                name="embedding_model",
                ok=True,
                latency_ms=0.0,
                detail=str(getattr(getattr(rag, "provider", None), "embedding_version", "")),
            )
        )
        deps.append(
            DependencyStatus(
                name="reranker",
                ok=True,
                latency_ms=0.0,
                detail=str(getattr(getattr(rag, "reranker", None), "name", "none")),
            )
        )
        deps.append(
            DependencyStatus(
                name="llm",
                ok=True,
                latency_ms=0.0,
                detail=getattr(getattr(rag, "llm", None), "model", "none (extractive)"),
            )
        )
    if state.ready_error:
        deps.append(
            DependencyStatus(name="startup", ok=False, latency_ms=0.0, detail=state.ready_error)
        )
    return deps


@router.get("", response_model=Liveness)
def live(state: AppState = Depends(get_state)) -> Liveness:
    return Liveness(status="ok", version=state.version, env=state.settings.core.env.value)


@router.get("/ready", response_model=Readiness)
def ready(response: Response, state: AppState = Depends(get_state)) -> Readiness:
    deps = check_dependencies(state)
    required = {"postgres", "minio", "startup"}
    failing = [
        d.name
        for d in deps
        if not d.ok
        and (d.name in required or d.name == "qdrant" and getattr(state.rag, "bm25", None) is None)
    ]
    ok = not failing and state.rag is not None
    response.status_code = 200 if ok else 503
    return Readiness(
        status="ok" if ok else "not_ready",
        ready=ok,
        version=state.version,
        embedding_version=getattr(getattr(state.rag, "provider", None), "embedding_version", None),
        failing=failing,
        dependencies=deps,
    )


@router.get("/dependencies", response_model=list[DependencyStatus])
def dependencies(state: AppState = Depends(get_state)) -> list[DependencyStatus]:
    return check_dependencies(state)
