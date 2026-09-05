"""akl_ingestion — discover, fetch to Bronze, parse/validate to Silver, quality gate, publish (PRD §7.3)."""

from __future__ import annotations

from typing import Any

import pendulum
from airflow.decorators import dag, task
from airflow.utils.trigger_rule import TriggerRule

from akl_airflow.common import AKL_PYTHON, dag_config, default_args
from akl_airflow.datasets import SILVER_DOCUMENTS

cfg = dag_config("ingestion")


@dag(
    dag_id=cfg.dag_id,
    schedule=cfg.schedule,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=cfg.max_active_runs,
    default_args=default_args(cfg),
    render_template_as_native_obj=True,
    params={"connectors": [], "force_reparse": False, "correlation_id": None},
    tags=["akl", "ingestion"],
    doc_md=__doc__,
)
def akl_ingestion() -> None:
    @task.external_python(
        python=AKL_PYTHON,
        expect_airflow=False,
        execution_timeout=cfg.timeout("load_connector_configs", 60),
        retries=1,
    )
    def load_connector_configs(akl_run_id: str, connectors: list[str] | None) -> list[str]:
        from akl.pipelines.airflow_tasks import list_connectors

        return list_connectors(akl_run_id, connectors or None)

    @task.external_python(
        python=AKL_PYTHON,
        expect_airflow=False,
        execution_timeout=cfg.timeout("fetch_to_bronze", 2700),
        pool="akl_ingestion",
        sla=None,
    )
    def fetch_to_bronze(akl_run_id: str, connector_id: str) -> dict[str, Any]:
        from akl.pipelines.airflow_tasks import fetch_connector

        return fetch_connector(akl_run_id, connector_id)

    @task.external_python(
        python=AKL_PYTHON,
        expect_airflow=False,
        execution_timeout=cfg.timeout("parse_validate_to_silver", 3600),
        retries=1,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )
    def parse_validate_to_silver(akl_run_id: str, limit: int) -> dict[str, Any]:
        from akl.pipelines.airflow_tasks import parse_backlog

        return parse_backlog(akl_run_id, limit=limit)

    @task(retries=0)
    def collect_fetch_reports(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Materialise the mapped XCom proxy before crossing into the akl venv (ADR-013)."""
        return [dict(r) for r in reports]

    @task.external_python(
        python=AKL_PYTHON,
        expect_airflow=False,
        execution_timeout=cfg.timeout("quality_gate", 120),
        retries=0,
    )
    def quality_gate(
        fetch_reports: list[dict[str, Any]], parse_report: dict[str, Any], max_ratio: float
    ) -> dict[str, Any]:
        from akl.pipelines.airflow_tasks import ingestion_gate

        return ingestion_gate(fetch_reports, parse_report, max_quarantine_ratio=max_ratio)

    @task(outlets=[SILVER_DOCUMENTS])
    def publish_silver_dataset(gate: dict[str, Any]) -> dict[str, Any]:
        return {"published": "akl://silver/documents", **gate}

    @task.external_python(
        python=AKL_PYTHON, expect_airflow=False, trigger_rule=TriggerRule.ALL_DONE, retries=0
    )
    def finalize(akl_run_id: str, gate: dict[str, Any] | None) -> dict[str, Any]:
        from akl.pipelines.airflow_tasks import finish_run

        out = finish_run(akl_run_id, "akl_ingestion", state="success" if gate else "failed")
        if out["state"] != "success":
            raise RuntimeError(
                "akl_ingestion: upstream task failed (see quality_gate / fetch logs)"
            )
        return out

    akl_run_id = "{{ run_id }}"
    connector_ids = load_connector_configs(akl_run_id, "{{ params.connectors }}")
    fetch_reports = fetch_to_bronze.partial(akl_run_id=akl_run_id).expand(
        connector_id=connector_ids
    )
    parsed = parse_validate_to_silver(akl_run_id, int(cfg.get("parse_limit", 500)))
    fetch_reports >> parsed
    collected = collect_fetch_reports(fetch_reports)
    gate = quality_gate(collected, parsed, float(cfg.gates.get("max_quarantine_ratio", 0.25)))
    publish_silver_dataset(gate) >> finalize(akl_run_id, gate)


akl_ingestion()
