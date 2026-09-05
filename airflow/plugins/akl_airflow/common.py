"""Shared DAG plumbing: YAML config, default_args, callbacks (PRD §7.1, §7.9).

Business logic never lives here or in DAG files — tasks call ``akl.pipelines.airflow_tasks``
inside an isolated interpreter (``AKL_PYTHON``) because Airflow 2.x pins SQLAlchemy 1.4 while
the ``akl`` package requires SQLAlchemy 2 (ADR-013).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("akl.airflow")

AKL_PYTHON = os.environ.get("AKL_PYTHON", "/opt/akl-venv/bin/python")
CONFIG_DIR = Path(os.environ.get("AKL_DAG_CONFIG_DIR", "/opt/airflow/configs/dags"))


@dataclass
class DagConfig:
    dag_id: str
    schedule: str | None
    max_active_runs: int = 1
    retries: int = 2
    retry_delay_s: int = 120
    max_retry_delay_s: int = 1200
    sla_minutes: int | None = None
    timeouts_s: dict[str, int] = field(default_factory=dict)
    gates: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def timeout(self, task_id: str, default_s: int = 900) -> timedelta:
        return timedelta(seconds=int(self.timeouts_s.get(task_id, default_s)))

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


def dag_config(name: str) -> DagConfig:
    """Load ``configs/dags/<name>.yaml``; missing file → safe defaults so DAG parsing never fails."""
    path = CONFIG_DIR / f"{name}.yaml"
    raw: dict[str, Any] = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw = loaded if isinstance(loaded, dict) else {}
    else:
        log.warning("dag config %s missing; using defaults", path)
    return DagConfig(
        dag_id=str(raw.get("dag_id", f"akl_{name}")),
        schedule=raw.get("schedule"),
        max_active_runs=int(raw.get("max_active_runs", 1)),
        retries=int(raw.get("retries", 2)),
        retry_delay_s=int(raw.get("retry_delay_s", 120)),
        max_retry_delay_s=int(raw.get("max_retry_delay_s", 1200)),
        sla_minutes=raw.get("sla_minutes"),
        timeouts_s={str(k): int(v) for k, v in (raw.get("timeouts_s") or {}).items()},
        gates={str(k): float(v) for k, v in (raw.get("gates") or {}).items()},
        raw=raw,
    )


def on_failure(context: dict[str, Any]) -> None:
    """Failure callback: structured log line (metrics push arrives with the observability batch)."""
    ti = context.get("task_instance")
    log.error(
        "akl_task_failed dag=%s task=%s run_id=%s try=%s exc=%s",
        getattr(ti, "dag_id", "?"),
        getattr(ti, "task_id", "?"),
        context.get("run_id"),
        getattr(ti, "try_number", "?"),
        context.get("exception"),
    )


def default_args(cfg: DagConfig) -> dict[str, Any]:
    args: dict[str, Any] = {
        "owner": "akl",
        "retries": cfg.retries,
        "retry_delay": timedelta(seconds=cfg.retry_delay_s),
        "retry_exponential_backoff": True,
        "max_retry_delay": timedelta(seconds=cfg.max_retry_delay_s),
        "on_failure_callback": on_failure,
        "depends_on_past": False,
    }
    if cfg.sla_minutes:
        args["sla"] = timedelta(minutes=int(cfg.sla_minutes))
    return args
