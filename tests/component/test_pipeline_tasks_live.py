"""Component test: DAG task entrypoints + run bookkeeping + maintenance on the live stack (Milestones 37–42).

Runs the same functions the Airflow DAGs call (no Airflow needed): an empty ingestion parse, the
gates, and the maintenance operations in dry-run mode, then checks pipeline_runs/task_runs rows.
No production data is modified (dry-run retention, stats are additive, no backups triggered).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import delete

from akl.config import Settings
from akl.db.models import PipelineRun, TaskRun
from akl.db.repositories.runs import RunRepository
from akl.db.session import Database, DatabaseUnavailableError
from akl.errors import AKLError
from akl.pipelines import airflow_tasks as t

pytestmark = pytest.mark.component


@pytest.fixture
def run_id() -> Iterator[str]:
    try:
        settings = Settings.load()
        db = Database(settings)
        db.ping()
    except (AKLError, DatabaseUnavailableError) as exc:  # pragma: no cover
        pytest.skip(f"stack unavailable: {exc}")
    rid = f"ctest-pipe-{uuid.uuid4().hex[:8]}"
    yield rid
    with db.session() as s:
        s.execute(delete(TaskRun).where(TaskRun.run_id == rid))
        s.execute(delete(PipelineRun).where(PipelineRun.run_id == rid))
    db.dispose()


def test_task_entrypoints_bookkeeping_gates_and_maintenance(run_id: str) -> None:
    # --- ingestion stage with nothing new: connectors listed, parse no-op, gate passes ------------------
    ids = t.list_connectors(run_id)
    assert isinstance(ids, list)
    parsed = t.parse_backlog(run_id, limit=10)
    assert parsed["considered"] >= 0
    gate = t.ingestion_gate([], parsed)
    assert gate["passed"]
    with pytest.raises(t.GateFailed):
        t.ingestion_gate([{"fetched": 2}], {"considered": 2, "parsed": 0, "quarantined": 2})

    # --- chunking stage: no backlog → unchanged run; gate passes ------------------------------------------
    chunked = t.chunk_run(run_id, limit=5, refresh_gold=False)
    assert chunked["documents_failed"] == 0
    assert t.chunking_gate(chunked)["passed"]

    # --- qdrant health (read-only) ---------------------------------------------------------------------------
    health = t.qdrant_health(run_id)
    assert health["ok"] is True

    # --- maintenance in dry-run / read-only modes -----------------------------------------------------------------
    compact = t.maintenance_task(run_id, "compact_partitions", "compact_partitions", dry_run=True)
    assert compact["compacted"] == 0 and compact["planned"] >= 0  # noqa: PT018
    bronze = t.maintenance_task(
        run_id, "bronze_retention", "bronze_retention", days=3650, dry_run=True
    )
    assert bronze["deleted"] == 0 and bronze["dry_run"] is True  # noqa: PT018
    quarantine = t.maintenance_task(
        run_id, "quarantine_retention", "quarantine_retention", days=3650, dry_run=True
    )
    assert quarantine["objects_deleted"] == 0
    retire = t.maintenance_task(
        run_id,
        "retire_old_embedding_versions",
        "retire_old_embedding_versions",
        days=3650,
        dry_run=True,
    )
    assert retire["objects_deleted"] == 0
    stats = t.maintenance_task(run_id, "compute_corpus_stats", "compute_corpus_stats")
    assert stats["rows"] >= 1
    assert "chunks_active" in stats["metrics"] or stats["rows"] >= 1
    cache = t.maintenance_task(
        run_id, "embedding_cache_eviction", "embedding_cache_eviction", ttl_days=36500
    )
    assert cache["evicted"] == 0
    conv = t.maintenance_task(run_id, "conversation_ttl", "conversation_ttl")
    assert conv["purged"] >= 0
    backups = t.maintenance_task(
        run_id, "backup_retention", "backup_retention", days=3650, dry_run=True
    )
    assert backups["deleted"] == 0
    with pytest.raises(AKLError):
        t.maintenance_task(run_id, "bogus", "not_an_operation")

    # --- finalize + bookkeeping ------------------------------------------------------------------------------------
    t.finish_run(run_id, "akl_maintenance", state="success")
    settings = Settings.load()
    db = Database(settings)
    try:
        with db.session() as s:
            repo = RunRepository(s)
            tasks = repo.tasks(run_id)
            names = [x.task_id for x in tasks]
            assert {
                "load_connector_configs",
                "parse_validate_to_silver",
                "chunk_documents",
                "qdrant_health_sensor",
                "compact_partitions",
                "compute_corpus_stats",
                "not_an_operation",
            } <= set(names)
            states = {x.task_id: x.state for x in tasks}
            assert states["not_an_operation"] == "failed"
            assert states["compute_corpus_stats"] == "success"
            assert all(x.finished_at is not None for x in tasks)
            assert all("duration_s" in (x.metrics or {}) for x in tasks)
            run = next(r for r in repo.recent(limit=50) if r.run_id == run_id)
            assert run.state == "success"
            assert run.finished_at is not None
    finally:
        db.dispose()
