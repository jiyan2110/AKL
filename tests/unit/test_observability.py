"""Unit tests: structured logging, metrics catalog, tracing, freshness (Milestones 43–48)."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta

import pytest

from akl.config import Environment, ObservabilitySettings, Settings
from akl.observability.freshness import evaluate_freshness
from akl.observability.logging import bind_context, configure_logging, get_logger, redact_processor
from akl.observability.metrics import PipelineMetrics, apply_task_metrics
from akl.observability.tracing import configure_tracing, get_tracer, traced

pytestmark = pytest.mark.unit


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    for key in ("AKL_DB_PASSWORD", "AKL_S3_ACCESS_KEY", "AKL_S3_SECRET_KEY"):
        monkeypatch.setenv(key, "x")
    return Settings.load(config_file=None, env_file=None)


# --------------------------------------------------------------------------- logging
def test_redact_processor_masks_secrets_tokens_and_url_creds() -> None:
    event = {
        "password": "hunter2",
        "AUTHORIZATION": "Bearer abc.def.ghi",
        "note": "url is postgresql://alice:s3cr3t@host:5432/db",
        "nested": {"api_key": "sk-live-123", "safe": "ok"},
        "items": [{"token": "tok-1"}, "plain-string"],
        "ok": True,
        "count": 3,
    }
    out = redact_processor(None, "info", dict(event))
    assert out["password"] == "***REDACTED***"
    assert out["AUTHORIZATION"] == "***REDACTED***"
    assert "***REDACTED***" in out["note"]
    assert "s3cr3t" not in out["note"] and "alice" not in out["note"]  # noqa: PT018
    assert out["nested"]["api_key"] == "***REDACTED***"
    assert out["nested"]["safe"] == "ok"
    assert out["items"][0]["token"] == "***REDACTED***"
    assert out["items"][1] == "plain-string"
    assert out["ok"] is True
    assert out["count"] == 3


def test_configure_logging_and_bind_context_dev_and_prod(settings: Settings) -> None:
    configure_logging(settings)  # dev: console renderer
    log = get_logger("test")
    buf = io.StringIO()
    with redirect_stdout(buf):
        with bind_context(request_id="r-1"):
            log.info("inside", extra_field=1)
        log.info("outside")
    output = buf.getvalue()
    assert "r-1" in output.split("\n")[0]
    assert "r-1" not in output.split("\n")[1]  # context reset after the `with` block

    prod = settings.model_copy(
        update={"core": settings.core.model_copy(update={"env": Environment.PROD})}
    )
    configure_logging(prod)  # prod: JSON renderer
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        get_logger("test2").info("json_line", password="nope")
    assert '"password": "***REDACTED***"' in buf2.getvalue()
    configure_logging(settings)  # restore dev renderer for other tests in this process


# --------------------------------------------------------------------------- metrics
def test_apply_task_metrics_populates_expected_gauges_and_counters() -> None:
    pm = PipelineMetrics()
    apply_task_metrics(
        pm,
        dag_id="akl_ingestion",
        task_id="fetch_to_bronze",
        out={"rows_in": 10, "rows_out": 8, "fetched": 8, "connector_id": "docs"},
    )
    assert (
        pm.task_rows_in.labels(dag_id="akl_ingestion", task_id="fetch_to_bronze")._value.get() == 10
    )
    assert (
        pm.task_rows_out.labels(dag_id="akl_ingestion", task_id="fetch_to_bronze")._value.get() == 8
    )
    assert pm.ingestion_fetched.labels(connector_id="docs")._value.get() == 8

    apply_task_metrics(
        pm,
        dag_id="akl_ingestion",
        task_id="parse_validate_to_silver",
        out={"quarantined": 2, "quarantine_ratio": 0.25},
    )
    assert pm.ingestion_quarantined._value.get() == 2
    assert pm.ingestion_quarantine_ratio._value.get() == 0.25

    apply_task_metrics(
        pm,
        dag_id="akl_chunking",
        task_id="chunk_documents",
        out={
            "documents_chunked": 5,
            "documents_unchanged": 1,
            "documents_failed": 0,
            "unchanged": 3,
            "modified": 2,
            "added": 1,
        },
    )
    assert pm.chunking_documents.labels(status="chunked")._value.get() == 5
    assert pm.chunking_ops.labels(kind="modified")._value.get() == 2

    apply_task_metrics(
        pm,
        dag_id="akl_embedding",
        task_id="embed_backlog",
        out={
            "coverage": 0.995,
            "backlog": 10,
            "remaining_backlog": 0,
            "cache_hits": 4,
            "throughput_cps": 12.5,
        },
    )
    assert pm.embedding_coverage._value.get() == 0.995
    assert pm.embedding_backlog._value.get() == 0
    assert pm.embedding_cache_hit_ratio._value.get() == 0.4
    assert pm.embedding_throughput._value.get() == 12.5

    apply_task_metrics(
        pm, dag_id="akl_qdrant_sync", task_id="reconcile", out={"drift": 0, "after": 100}
    )
    assert pm.qdrant_drift._value.get() == 0
    assert pm.qdrant_points._value.get() == 100

    apply_task_metrics(
        pm,
        dag_id="akl_qdrant_sync",
        task_id="rebuild_bm25_index",
        out={"documents": 100, "terms": 500},
    )
    assert pm.bm25_documents._value.get() == 100

    apply_task_metrics(
        pm,
        dag_id="akl_maintenance",
        task_id="compact_partitions",
        out={"bytes_before": 1000, "bytes_after": 200},
    )
    assert pm.maintenance_bytes_reclaimed.labels(operation="compact_partitions")._value.get() == 800

    # unknown keys are ignored, not an error
    apply_task_metrics(
        pm, dag_id="akl_maintenance", task_id="vacuum_analyze", out={"tables": ["documents"]}
    )


def test_pipeline_metrics_push_is_a_noop_without_a_gateway_url() -> None:
    pm = PipelineMetrics()
    pm.push("", job="akl_pipelines", dag_id="akl_ingestion", task_id="x")  # must not raise
    pm.push(
        "http://127.0.0.1:1", job="akl_pipelines", dag_id="akl_ingestion", task_id="x"
    )  # unreachable: swallowed


# --------------------------------------------------------------------------- tracing
def test_traced_is_a_true_noop_when_otel_disabled(settings: Settings) -> None:
    assert settings.observability.otel_enabled is False
    configure_tracing(settings)
    with traced("unit.test.span", foo="bar", n=1) as span:
        assert span is not None
    tracer = get_tracer()
    assert tracer is not None


# --------------------------------------------------------------------------- freshness
def test_evaluate_freshness_fresh_stale_and_never_run() -> None:
    now = datetime(2026, 9, 6, 12, 0, tzinfo=UTC)
    fresh = evaluate_freshness("akl_ingestion", now - timedelta(minutes=10), 60, now=now)
    assert fresh.stale is False
    assert fresh.age_minutes == pytest.approx(10.0)

    stale = evaluate_freshness("akl_embedding", now - timedelta(minutes=200), 120, now=now)
    assert stale.stale is True
    assert stale.age_minutes == pytest.approx(200.0)

    never = evaluate_freshness("akl_qdrant_sync", None, 30, now=now)
    assert never.stale is True
    assert never.age_minutes is None


def test_observability_settings_defaults_are_safe() -> None:
    s = ObservabilitySettings()
    assert s.pushgateway_url == ""
    assert s.otel_enabled is False
    assert s.mlflow_enabled is False
    assert s.lineage_enabled is True
    assert s.freshness_stale_after_minutes["akl_ingestion"] == 60
