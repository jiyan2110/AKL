"""akl_qdrant_sync — health check, reconcile Gold→Qdrant, verify drift, rebuild BM25, notify API (PRD §7.6)."""

from __future__ import annotations

from typing import Any

import pendulum
from airflow.decorators import dag, task
from airflow.timetables.datasets import DatasetOrTimeSchedule
from airflow.timetables.trigger import CronTriggerTimetable
from airflow.utils.trigger_rule import TriggerRule

from akl_airflow.common import AKL_PYTHON, dag_config, default_args
from akl_airflow.datasets import GOLD_EMBEDDINGS

cfg = dag_config("qdrant_sync")
schedule = DatasetOrTimeSchedule(
    timetable=CronTriggerTimetable(cfg.schedule or "45 * * * *", timezone="UTC"),
    datasets=[GOLD_EMBEDDINGS],
)


@dag(
    dag_id=cfg.dag_id,
    schedule=schedule,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=cfg.max_active_runs,
    default_args=default_args(cfg),
    render_template_as_native_obj=True,
    params={"dry_run": False},
    tags=["akl", "vector"],
    doc_md=__doc__,
)
def akl_qdrant_sync() -> None:
    @task.external_python(
        python=AKL_PYTHON,
        expect_airflow=False,
        execution_timeout=cfg.timeout("qdrant_health_sensor", 600),
        retries=5,
    )
    def qdrant_health_sensor(akl_run_id: str) -> dict[str, Any]:
        from akl.pipelines.airflow_tasks import qdrant_health

        return qdrant_health(akl_run_id)

    @task.external_python(
        python=AKL_PYTHON, expect_airflow=False, execution_timeout=cfg.timeout("reconcile", 2700)
    )
    def reconcile(akl_run_id: str, dry_run: bool) -> dict[str, Any]:
        from akl.pipelines.airflow_tasks import qdrant_sync

        return qdrant_sync(akl_run_id, dry_run=bool(dry_run))

    @task.external_python(
        python=AKL_PYTHON,
        expect_airflow=False,
        execution_timeout=cfg.timeout("rebuild_bm25_index", 1200),
    )
    def rebuild_bm25_index(akl_run_id: str, sync_report: dict[str, Any]) -> dict[str, Any]:
        from akl.pipelines.airflow_tasks import bm25_build

        if sync_report.get("drift", 0) != 0:
            raise RuntimeError(
                f"refusing to rebuild BM25 while Qdrant drift is {sync_report.get('drift')}"
            )
        return bm25_build(akl_run_id)

    @task.external_python(
        python=AKL_PYTHON,
        expect_airflow=False,
        execution_timeout=cfg.timeout("notify_api_reload", 120),
        retries=3,
    )
    def notify_api_reload(
        akl_run_id: str, api_url: str, bm25_report: dict[str, Any]
    ) -> dict[str, Any]:
        from akl.pipelines.airflow_tasks import notify_api_reload as notify

        return notify(akl_run_id, api_url=api_url or None)

    @task.external_python(
        python=AKL_PYTHON, expect_airflow=False, trigger_rule=TriggerRule.ALL_DONE, retries=0
    )
    def finalize(akl_run_id: str, sync_report: dict[str, Any] | None) -> dict[str, Any]:
        from akl.pipelines.airflow_tasks import finish_run

        ok = bool(sync_report) and sync_report.get("drift", 1) == 0
        out = finish_run(akl_run_id, "akl_qdrant_sync", state="success" if ok else "failed")
        if out["state"] != "success":
            raise RuntimeError("akl_qdrant_sync: upstream task failed (see task logs)")
        return out

    akl_run_id = "{{ run_id }}"
    health = qdrant_health_sensor(akl_run_id)
    sync_report = reconcile(akl_run_id, "{{ params.dry_run }}")
    health >> sync_report
    bm25 = rebuild_bm25_index(akl_run_id, sync_report)
    notify_api_reload(akl_run_id, str(cfg.get("api_url", "") or ""), bm25) >> finalize(
        akl_run_id, sync_report
    )


akl_qdrant_sync()
