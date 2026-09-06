"""Freshness (PRD §8.6): how long since each DAG last succeeded, and whether that is too long.

``refresh_freshness_gauges`` is called by the API's readiness/dependencies check and by
``akl-cli pipeline freshness``; it sets the scraped ``akl_dag_last_success_timestamp_seconds`` /
``akl_dag_stale`` gauges directly from ``pipeline_runs`` — no separate exporter process needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from akl.config import ObservabilitySettings
from akl.db.repositories.runs import RunRepository
from akl.db.session import Database
from akl.observability.metrics import DAG_LAST_SUCCESS_TIMESTAMP, DAG_STALE

DAGS: tuple[str, ...] = ("akl_ingestion", "akl_chunking", "akl_embedding", "akl_qdrant_sync")


@dataclass(frozen=True)
class Freshness:
    dag_id: str
    last_success_at: datetime | None
    age_minutes: float | None
    stale_after_minutes: int
    stale: bool


def evaluate_freshness(
    dag_id: str, last_success_at: datetime | None, stale_after_minutes: int, *, now: datetime
) -> Freshness:
    """Pure evaluation: age in minutes and whether it exceeds the threshold (unit-testable in isolation)."""
    age = (now - last_success_at).total_seconds() / 60.0 if last_success_at else None
    return Freshness(
        dag_id,
        last_success_at,
        age,
        stale_after_minutes,
        stale=age is None or age > stale_after_minutes,
    )


def check_freshness(
    db: Database,
    settings: ObservabilitySettings,
    *,
    dags: tuple[str, ...] = DAGS,
    now: datetime | None = None,
) -> list[Freshness]:
    now = now or datetime.now(UTC)
    out: list[Freshness] = []
    with db.session() as s:
        repo = RunRepository(s)
        for dag_id in dags:
            last = repo.last_success(dag_id)
            threshold = int(settings.freshness_stale_after_minutes.get(dag_id, 120))
            out.append(evaluate_freshness(dag_id, last, threshold, now=now))
    return out


def refresh_freshness_gauges(
    db: Database, settings: ObservabilitySettings, *, dags: tuple[str, ...] = DAGS
) -> list[Freshness]:
    results = check_freshness(db, settings, dags=dags)
    for r in results:
        DAG_LAST_SUCCESS_TIMESTAMP.labels(dag_id=r.dag_id).set(
            r.last_success_at.timestamp() if r.last_success_at else 0.0
        )
        DAG_STALE.labels(dag_id=r.dag_id).set(1.0 if r.stale else 0.0)
    return results
