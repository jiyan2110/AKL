"""Component test: DAG task entrypoints + run bookkeeping + maintenance + observability (M37-M48).

Runs the same functions the Airflow DAGs call (no Airflow needed): an empty ingestion parse, the
gates, and the maintenance operations in dry-run mode, then checks pipeline_runs/task_runs rows,
lineage_edges (Batch G), and DAG freshness gauges (Batch G). No production data is modified
(dry-run retention, stats are additive, no backups triggered).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import delete

from akl.config import Settings
from akl.db.models import LineageEdge, PipelineRun, TaskRun
from akl.db.repositories.lineage import LineageRepository
from akl.db.repositories.runs import RunRepository
from akl.db.session import Database, DatabaseUnavailableError
from akl.errors import AKLError
from akl.observability.freshness import refresh_freshness_gauges
from akl.observability.metrics import DAG_LAST_SUCCESS_TIMESTAMP, DAG_STALE
from akl.pipelines import airflow_tasks as t

pytestmark = pytest.mark.component


@pytest.fixture
def tag() -> Iterator[str]:
    try:
        settings = Settings.load()
        db = Database(settings)
        db.ping()
    except (AKLError, DatabaseUnavailableError) as exc:  # pragma: no cover
        pytest.skip(f"stack unavailable: {exc}")
    value = uuid.uuid4().hex[:8]
    yield value
    prefix = f"ctest-pipe-{value}"
    with db.session() as s:
        s.execute(delete(TaskRun).where(TaskRun.run_id.like(f"{prefix}%")))
        s.execute(delete(LineageEdge).where(LineageEdge.run_id.like(f"{prefix}%")))
        s.execute(delete(PipelineRun).where(PipelineRun.run_id.like(f"{prefix}%")))
    db.dispose()


def test_task_entrypoints_bookkeeping_gates_and_maintenance(tag: str) -> None:
    run_id = f"ctest-pipe-{tag}"
    # Each DAG gets its own run_id below: `pipeline_runs` is keyed by run_id with one `dag_id`
    # column, so reusing one run_id across DAGs would overwrite it on every task_scope() call
    # (harmless for the bookkeeping assertions, but it would break the freshness check, which
    # looks up "last successful run of dag X" by dag_id).
    ingestion_run_id = f"{run_id}-ing"
    chunking_run_id = f"{run_id}-chunk"

    # --- ingestion stage with nothing new: connectors listed, parse no-op, gate passes ------------------
    ids = t.list_connectors(ingestion_run_id)
    assert isinstance(ids, list)
    parsed = t.parse_backlog(ingestion_run_id, limit=10)
    assert parsed["considered"] >= 0
    gate = t.ingestion_gate([], parsed)
    assert gate["passed"]
    with pytest.raises(t.GateFailed):
        t.ingestion_gate([{"fetched": 2}], {"considered": 2, "parsed": 0, "quarantined": 2})
    t.finish_run(ingestion_run_id, "akl_ingestion", state="success")

    # --- chunking stage: no backlog → unchanged run; gate passes ------------------------------------------
    chunked = t.chunk_run(chunking_run_id, limit=5, refresh_gold=False)
    assert chunked["documents_failed"] == 0
    assert t.chunking_gate(chunked)["passed"]
    t.finish_run(chunking_run_id, "akl_chunking", state="success")

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
                "qdrant_health_sensor",
                "compact_partitions",
                "compute_corpus_stats",
                "not_an_operation",
            } <= set(names)
            assert {x.task_id for x in repo.tasks(ingestion_run_id)} == {
                "load_connector_configs",
                "parse_validate_to_silver",
            }
            assert {x.task_id for x in repo.tasks(chunking_run_id)} == {"chunk_documents"}
            states = {x.task_id: x.state for x in tasks}
            assert states["not_an_operation"] == "failed"
            assert states["compute_corpus_stats"] == "success"
            assert all(x.finished_at is not None for x in tasks)
            assert all("duration_s" in (x.metrics or {}) for x in tasks)
            run = next(r for r in repo.recent(limit=50) if r.run_id == run_id)
            assert run.state == "success"
            assert run.finished_at is not None

            # --- lineage: dataset edges recorded at the two stage boundaries this run touched -----------
            lineage = LineageRepository(s)
            ingestion_edges = {e.output_dataset: e for e in lineage.for_run(ingestion_run_id)}
            assert "silver/documents" in ingestion_edges
            assert ingestion_edges["silver/documents"].input_dataset == "bronze/manifest"
            assert ingestion_edges["silver/documents"].rows_out == parsed["parsed"]
            chunking_edges = {e.output_dataset: e for e in lineage.for_run(chunking_run_id)}
            assert "silver/chunks" in chunking_edges
            assert chunking_edges["silver/chunks"].input_dataset == "silver/documents"
            recent_for_dataset = lineage.for_dataset("silver/chunks", limit=5)
            assert chunking_run_id in {e.run_id for e in recent_for_dataset}

        # --- freshness: ingestion/chunking just succeeded, so neither should be stale ----------------------
        results = refresh_freshness_gauges(
            db, settings.observability, dags=("akl_ingestion", "akl_chunking")
        )
        by_dag = {r.dag_id: r for r in results}
        assert by_dag["akl_ingestion"].stale is False
        assert by_dag["akl_chunking"].stale is False
        assert by_dag["akl_ingestion"].age_minutes is not None
        assert by_dag["akl_ingestion"].age_minutes < 5
        assert DAG_STALE.labels(dag_id="akl_ingestion")._value.get() == 0.0
        assert DAG_LAST_SUCCESS_TIMESTAMP.labels(dag_id="akl_ingestion")._value.get() > 0
    finally:
        db.dispose()
