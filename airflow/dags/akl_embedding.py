"""akl_embedding — warm model, embed Gold backlog (cache-aware), coverage gate, publish (PRD §7.5)."""

from __future__ import annotations

from typing import Any

import pendulum
from airflow.decorators import dag, task
from airflow.timetables.datasets import DatasetOrTimeSchedule
from airflow.timetables.trigger import CronTriggerTimetable
from airflow.utils.trigger_rule import TriggerRule

from akl_airflow.common import AKL_PYTHON, dag_config, default_args
from akl_airflow.datasets import GOLD_EMBEDDINGS, SILVER_CHUNKS

cfg = dag_config("embedding")
schedule = DatasetOrTimeSchedule(
    timetable=CronTriggerTimetable(cfg.schedule or "30 * * * *", timezone="UTC"),
    datasets=[SILVER_CHUNKS],
)


@dag(
    dag_id=cfg.dag_id,
    schedule=schedule,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=cfg.max_active_runs,
    default_args=default_args(cfg),
    render_template_as_native_obj=True,
    params={"limit": None},
    tags=["akl", "embedding"],
    doc_md=__doc__,
)
def akl_embedding() -> None:
    @task.external_python(
        python=AKL_PYTHON,
        expect_airflow=False,
        execution_timeout=cfg.timeout("warm_model_check", 600),
        retries=3,
    )
    def warm_model_check(akl_run_id: str) -> dict[str, Any]:
        from akl.pipelines.airflow_tasks import warm_model

        return warm_model(akl_run_id)

    @task.external_python(
        python=AKL_PYTHON,
        expect_airflow=False,
        execution_timeout=cfg.timeout("embed_backlog", 5400),
        pool=str(cfg.get("pool", "akl_embedding")),
    )
    def embed_backlog(akl_run_id: str, limit: int | None) -> dict[str, Any]:
        from akl.pipelines.airflow_tasks import embed_run

        return embed_run(akl_run_id, limit=limit)

    @task.external_python(
        python=AKL_PYTHON,
        expect_airflow=False,
        execution_timeout=cfg.timeout("coverage_gate", 120),
        retries=0,
    )
    def coverage_gate(report: dict[str, Any], min_coverage: float) -> dict[str, Any]:
        from akl.pipelines.airflow_tasks import coverage_gate as gate

        return gate(report, min_coverage=min_coverage)

    @task(outlets=[GOLD_EMBEDDINGS])
    def publish_gold_embeddings(report: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
        return {
            "published": "akl://gold/chunk_embeddings",
            "written": report.get("written"),
            **gate,
        }

    @task.external_python(
        python=AKL_PYTHON, expect_airflow=False, trigger_rule=TriggerRule.ALL_DONE, retries=0
    )
    def finalize(akl_run_id: str, report: dict[str, Any] | None) -> dict[str, Any]:
        from akl.pipelines.airflow_tasks import finish_run

        out = finish_run(akl_run_id, "akl_embedding", state="success" if report else "failed")
        if out["state"] != "success":
            raise RuntimeError("akl_qdrant_sync: upstream task failed (see task logs)")
        return out

    akl_run_id = "{{ run_id }}"
    warm = warm_model_check(akl_run_id)
    report = embed_backlog(akl_run_id, "{{ params.limit }}")
    warm >> report
    gate = coverage_gate(report, float(cfg.gates.get("min_coverage", 0.99)))
    publish_gold_embeddings(report, gate) >> finalize(akl_run_id, report)


akl_embedding()
