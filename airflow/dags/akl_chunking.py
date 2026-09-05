"""akl_chunking — chunk current documents lacking chunks, gate, refresh Gold, publish (PRD §7.4)."""

from __future__ import annotations

from typing import Any

import pendulum
from airflow.decorators import dag, task
from airflow.timetables.datasets import DatasetOrTimeSchedule
from airflow.timetables.trigger import CronTriggerTimetable
from airflow.utils.trigger_rule import TriggerRule

from akl_airflow.common import AKL_PYTHON, dag_config, default_args
from akl_airflow.datasets import SILVER_CHUNKS, SILVER_DOCUMENTS

cfg = dag_config("chunking")
schedule = DatasetOrTimeSchedule(
    timetable=CronTriggerTimetable(cfg.schedule or "15 * * * *", timezone="UTC"),
    datasets=[SILVER_DOCUMENTS],
)


@dag(
    dag_id=cfg.dag_id,
    schedule=schedule,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=cfg.max_active_runs,
    default_args=default_args(cfg),
    render_template_as_native_obj=True,
    params={"document_ids": [], "force_rechunk": False},
    tags=["akl", "chunking"],
    doc_md=__doc__,
)
def akl_chunking() -> None:
    @task.external_python(
        python=AKL_PYTHON,
        expect_airflow=False,
        execution_timeout=cfg.timeout("chunk_documents", 2700),
    )
    def chunk_documents(
        akl_run_id: str, limit: int, document_ids: list[str] | None
    ) -> dict[str, Any]:
        from akl.pipelines.airflow_tasks import chunk_run

        return chunk_run(
            akl_run_id, limit=limit, document_ids=document_ids or None, refresh_gold=True
        )

    @task.external_python(
        python=AKL_PYTHON,
        expect_airflow=False,
        execution_timeout=cfg.timeout("quality_gate", 120),
        retries=0,
    )
    def quality_gate(report: dict[str, Any], max_failed_ratio: float) -> dict[str, Any]:
        from akl.pipelines.airflow_tasks import chunking_gate

        return chunking_gate(report, max_failed_ratio=max_failed_ratio)

    @task(outlets=[SILVER_CHUNKS])
    def publish_silver_chunks(report: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
        return {
            "published": "akl://silver/chunks",
            "gold_snapshot_id": report.get("gold_snapshot_id"),
            **gate,
        }

    @task.external_python(
        python=AKL_PYTHON, expect_airflow=False, trigger_rule=TriggerRule.ALL_DONE, retries=0
    )
    def finalize(akl_run_id: str, report: dict[str, Any] | None) -> dict[str, Any]:
        from akl.pipelines.airflow_tasks import finish_run

        out = finish_run(
            akl_run_id,
            "akl_chunking",
            state="success" if report else "failed",
            gold_snapshot_id=(report or {}).get("gold_snapshot_id"),
        )
        if out["state"] != "success":
            raise RuntimeError("akl_chunking: upstream task failed (see task logs)")
        return out

    akl_run_id = "{{ run_id }}"
    report = chunk_documents(akl_run_id, int(cfg.get("limit", 200)), "{{ params.document_ids }}")
    gate = quality_gate(report, float(cfg.gates.get("max_failed_ratio", 0.3)))
    publish_silver_chunks(report, gate) >> finalize(akl_run_id, report)


akl_chunking()
