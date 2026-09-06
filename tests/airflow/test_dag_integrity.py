"""DAG integrity (PRD §11.3): every DAG imports, has retries/timeouts/owner, and the expected outlets.

Runs wherever Airflow is importable — the scheduler container (`make dags-test`) or CI — and is
skipped in the host venv (Airflow does not run on Windows).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

airflow = pytest.importorskip("airflow.models")  # the repo's airflow/ folder is not the package
pytestmark = pytest.mark.airflow

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "akl_ingestion": (
        "load_connector_configs",
        "fetch_to_bronze",
        "parse_validate_to_silver",
        "collect_fetch_reports",
        "quality_gate",
        "publish_silver_dataset",
        "finalize",
    ),
    "akl_chunking": ("chunk_documents", "quality_gate", "publish_silver_chunks", "finalize"),
    "akl_embedding": (
        "warm_model_check",
        "embed_backlog",
        "coverage_gate",
        "publish_gold_embeddings",
        "finalize",
    ),
    "akl_qdrant_sync": (
        "qdrant_health_sensor",
        "reconcile",
        "rebuild_bm25_index",
        "notify_api_reload",
        "finalize",
    ),
    "akl_maintenance": (
        "compact_partitions",
        "bronze_retention",
        "quarantine_retention",
        "embedding_cache_eviction",
        "conversation_ttl",
        "retire_old_embedding_versions",
        "compute_corpus_stats",
        "backup_postgres",
        "qdrant_snapshot",
        "backup_retention",
        "audit_log_retention",
        "vacuum_analyze",
        "finalize",
    ),
}
OUTLETS = {
    "akl_ingestion": {"akl://silver/documents"},
    "akl_chunking": {"akl://silver/chunks"},
    "akl_embedding": {"akl://gold/chunk_embeddings"},
}


@pytest.fixture(scope="module")
def dagbag():  # type: ignore[no-untyped-def]
    os.environ.setdefault("AKL_DAG_CONFIG_DIR", str(ROOT / "configs" / "dags"))
    sys.path.insert(0, str(ROOT / "airflow" / "plugins"))
    from airflow.models import DagBag

    return DagBag(dag_folder=str(ROOT / "airflow" / "dags"), include_examples=False)


def test_no_import_errors(dagbag) -> None:  # type: ignore[no-untyped-def]
    assert dagbag.import_errors == {}
    assert set(dagbag.dags) == set(EXPECTED)


@pytest.mark.parametrize("dag_id", sorted(EXPECTED))
def test_dag_contract(dagbag, dag_id: str) -> None:  # type: ignore[no-untyped-def]
    dag = dagbag.dags[dag_id]
    assert [t.task_id for t in dag.tasks] == list(EXPECTED[dag_id])
    assert dag.max_active_runs == 1
    assert dag.catchup is False
    assert {"akl"} <= set(dag.tags)
    for t in dag.tasks:
        assert t.owner == "akl"
        assert t.retries is not None
        if not t.task_id.startswith(("publish_", "finalize")):
            assert t.execution_timeout is not None, (dag_id, t.task_id)
    outlets = {d.uri for t in dag.tasks for d in getattr(t, "outlets", [])}
    assert outlets == OUTLETS.get(dag_id, set())
    dag.test_cycle() if hasattr(dag, "test_cycle") else None
