"""Dependency injection for the API (PRD §10.11 deps.py).

``AppState`` holds the long-lived objects (settings, DuckDB engine, Database, RAGService,
authenticator, rate limiter, background job registry). Tests construct their own ``AppState``
with fakes and pass it to ``create_app``.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, Header, Request

from akl.api import metrics
from akl.api.middleware.ratelimit import RateLimitedError, TokenBucketLimiter
from akl.config import Settings
from akl.db.session import Database
from akl.lakehouse.engine import DuckDBEngine
from akl.security.auth import Authenticator, AuthError, ForbiddenError, require_scope
from akl.security.principal import Principal


@dataclass
class Job:
    run_id: str
    state: str = "queued"
    stages: list[str] = field(default_factory=list)
    report: dict[str, Any] | None = None
    errors: list[dict[str, str]] = field(default_factory=list)


class JobRegistry:
    """In-process registry of background pipeline runs (surfaced at /v1/jobs/{run_id})."""

    def __init__(self, capacity: int = 200) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._capacity = capacity

    def create(self, run_id: str) -> Job:
        with self._lock:
            job = Job(run_id=run_id)
            self._jobs[run_id] = job
            if len(self._jobs) > self._capacity:
                oldest = next(iter(self._jobs))
                self._jobs.pop(oldest, None)
            return job

    def get(self, run_id: str) -> Job | None:
        return self._jobs.get(run_id)

    def update(self, run_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(run_id)
            if job is not None:
                for k, v in fields.items():
                    setattr(job, k, v)


@dataclass
class AppState:
    settings: Settings
    engine: DuckDBEngine | None
    db: Database | None
    rag: Any  # RAGService or a test double exposing search()/answer()/stream_answer()
    authenticator: Authenticator
    limiter: TokenBucketLimiter
    jobs: JobRegistry = field(default_factory=JobRegistry)
    version: str = "0.0.0"
    ready_error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)  # DuckDB engine is single-threaded


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.akl
    return state


def get_settings(state: AppState = Depends(get_state)) -> Settings:
    return state.settings


def get_request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", uuid.uuid4().hex))


def get_principal(
    state: AppState = Depends(get_state),
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> Principal:
    try:
        return state.authenticator.authenticate(authorization=authorization, api_key=x_api_key)
    except AuthError as exc:
        metrics.AUTH_FAILURES.labels(reason=exc.code).inc()
        raise


def scoped(scope: str) -> Any:
    def _dep(principal: Principal = Depends(get_principal)) -> Principal:
        try:
            require_scope(principal, scope)
        except ForbiddenError:
            metrics.AUTH_FAILURES.labels(reason="AKL-E1003").inc()
            raise
        return principal

    return _dep


def rate_limited(route_class: str) -> Any:
    def _dep(
        request: Request,
        state: AppState = Depends(get_state),
        principal: Principal = Depends(get_principal),
    ) -> None:
        allowed, retry_after = state.limiter.check(principal.subject, route_class)
        if not allowed:
            metrics.RATE_LIMITED.labels(route_class=route_class).inc()
            raise RateLimitedError(
                "rate limit exceeded",
                details={"retry_after_s": retry_after, "route_class": route_class},
            )

    return _dep
