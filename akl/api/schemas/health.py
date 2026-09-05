"""Health models (PRD §10.7)."""

from __future__ import annotations

from akl.api.schemas.common import StrictModel


class Liveness(StrictModel):
    status: str
    version: str
    env: str


class DependencyStatus(StrictModel):
    name: str
    ok: bool
    latency_ms: float
    detail: str = ""


class Readiness(StrictModel):
    status: str
    ready: bool
    version: str
    embedding_version: str | None
    failing: list[str]
    dependencies: list[DependencyStatus]
