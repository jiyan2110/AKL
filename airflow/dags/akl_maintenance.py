"""akl_maintenance — compaction, retention, cache eviction, TTLs, stats, backups, vacuum (PRD §7.7)."""

from __future__ import annotations

from typing import Any

import pendulum
from airflow.decorators import dag, task
from airflow.utils.trigger_rule import TriggerRule

from akl_airflow.common import AKL_PYTHON, dag_config, default_args

cfg = dag_config("maintenance")
retention = cfg.get("retention_days", {}) or {}
dry_run = bool(cfg.get("dry_run", False))


@dag(
    dag_id=cfg.dag_id,
    schedule=cfg.schedule,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=cfg.max_active_runs,
    default_args=default_args(cfg),
    render_template_as_native_obj=True,
    params={"dry_run": dry_run, "skip": []},
    tags=["akl", "maintenance"],
    doc_md=__doc__,
)
def akl_maintenance() -> None:
    def op(task_id: str, operation: str, timeout_key: str, **kwargs: Any) -> Any:
        """One maintenance task. Everything the body needs is passed as arguments: external_python
        ships only the function source, so closure variables would not exist in the akl venv."""

        @task.external_python(
            python=AKL_PYTHON,
            expect_airflow=False,
            task_id=task_id,
            execution_timeout=cfg.timeout(timeout_key, 900),
            retries=1,
        )
        def _run(
            akl_run_id: str, dry: bool, task_name: str, op_name: str, params: dict[str, Any]
        ) -> dict[str, Any]:
            from akl.pipelines.airflow_tasks import maintenance_task

            dry_aware = (
                "compact_partitions",
                "bronze_retention",
                "quarantine_retention",
                "retire_old_embedding_versions",
                "backup_retention",
            )
            if "dry_run" not in params and op_name in dry_aware:
                params = {**params, "dry_run": bool(dry)}
            return maintenance_task(akl_run_id, op_name, task_id=task_name, **params)

        def bound(akl_run_id: str, dry: Any) -> Any:
            return _run(akl_run_id, dry, task_id, operation, dict(kwargs))

        return bound

    akl_run_id = "{{ run_id }}"
    dry = "{{ params.dry_run }}"

    compact = op("compact_partitions", "compact_partitions", "compact_partitions")(akl_run_id, dry)
    bronze = op(
        "bronze_retention", "bronze_retention", "retention", days=int(retention.get("bronze", 365))
    )(akl_run_id, dry)
    quarantine = op(
        "quarantine_retention",
        "quarantine_retention",
        "retention",
        days=int(retention.get("quarantine", 90)),
    )(akl_run_id, dry)
    cache = op("embedding_cache_eviction", "embedding_cache_eviction", "retention")(akl_run_id, dry)
    conversations = op("conversation_ttl", "conversation_ttl", "retention")(akl_run_id, dry)
    retire = op(
        "retire_old_embedding_versions",
        "retire_old_embedding_versions",
        "retention",
        days=int(retention.get("embedding_versions", 30)),
    )(akl_run_id, dry)
    stats = op("compute_corpus_stats", "compute_corpus_stats", "compute_corpus_stats")(
        akl_run_id, dry
    )
    backup_pg = op("backup_postgres", "backup_postgres", "backup")(akl_run_id, dry)
    snapshot = op("qdrant_snapshot", "qdrant_snapshot", "backup")(akl_run_id, dry)
    backup_ret = op(
        "backup_retention", "backup_retention", "retention", days=int(retention.get("backups", 14))
    )(akl_run_id, dry)
    vacuum = op("vacuum_analyze", "vacuum_analyze", "vacuum_analyze")(akl_run_id, dry)

    @task.external_python(
        python=AKL_PYTHON, expect_airflow=False, trigger_rule=TriggerRule.ALL_DONE, retries=0
    )
    def finalize(akl_run_id: str, results: list[dict[str, Any] | None]) -> dict[str, Any]:
        from akl.pipelines.airflow_tasks import finish_run

        out = finish_run(
            akl_run_id, "akl_maintenance", state="success" if all(results) else "failed"
        )
        if out["state"] != "success":
            raise RuntimeError("akl_maintenance: upstream task failed (see task logs)")
        return out

    compact >> stats
    [backup_pg, snapshot] >> backup_ret
    [bronze, quarantine, cache, conversations, retire, stats, backup_ret] >> vacuum
    finalize(
        akl_run_id,
        [
            compact,
            bronze,
            quarantine,
            cache,
            conversations,
            retire,
            stats,
            backup_pg,
            snapshot,
            backup_ret,
            vacuum,
        ],
    )


akl_maintenance()
