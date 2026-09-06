"""MLflow logging for embedding runs (PRD §8.5). Disabled by default; failures never break a run.

Only the embedding pipeline logs to MLflow in this batch (it is the one stage with meaningful
params/metrics per run — model, batch size, cache hit rate, throughput). Retrieval evaluation
runs get their own MLflow experiment when the eval harness lands (Batch I).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from akl.config import ObservabilitySettings
from akl.observability.logging import get_logger

log = get_logger("akl.observability.mlflow")


@contextlib.contextmanager
def mlflow_run(
    settings: ObservabilitySettings, *, run_name: str, tags: dict[str, str] | None = None
) -> Iterator[Any]:
    """Yields an MLflow run context, or ``None`` when disabled/unreachable — never raises."""
    if not settings.mlflow_enabled:
        yield None
        return
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment)
        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            yield run
    except Exception as exc:  # pragma: no cover - MLflow server is optional infra
        log.warning("mlflow_unavailable", error=str(exc))
        yield None


def log_params(run: Any, params: dict[str, Any]) -> None:
    if run is None:
        return
    with contextlib.suppress(Exception):
        import mlflow

        mlflow.log_params({k: v for k, v in params.items() if v is not None})


def log_metrics(run: Any, metrics: dict[str, float | int]) -> None:
    if run is None:
        return
    with contextlib.suppress(Exception):
        import mlflow

        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
