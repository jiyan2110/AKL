"""Unit tests: quality gates, compaction planning, DAG config loading, maintenance helpers (Milestones 37–42)."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from akl.lakehouse.compaction import COMPACTABLE, group_by_partition, plan_partitions
from akl.lakehouse.io import FileInfo, Layer
from akl.pipelines.airflow_tasks import (
    GateFailed,
    _int_or_none,
    chunking_gate,
    coverage_gate,
    ingestion_gate,
)
from akl.pipelines.maintenance import compaction_layers

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_ingestion_gate_rules() -> None:
    ok = ingestion_gate(
        [{"fetched": 10}, {"fetched": 5}], {"considered": 15, "parsed": 13, "quarantined": 2}
    )
    assert ok["passed"]
    assert ok["quarantine_ratio"] == pytest.approx(2 / 15, abs=1e-4)
    assert ingestion_gate([], {"considered": 0, "parsed": 0, "quarantined": 0})[
        "passed"
    ]  # nothing to do is fine
    with pytest.raises(GateFailed) as exc:
        ingestion_gate(
            [{"fetched": 4}],
            {"considered": 4, "parsed": 2, "quarantined": 2},
            max_quarantine_ratio=0.25,
        )
    assert exc.value.code == "AKL-E7001"
    assert exc.value.details["quarantine_ratio"] == 0.5
    with pytest.raises(GateFailed):
        ingestion_gate([{"fetched": 3}], {"considered": 3, "parsed": 0, "quarantined": 0})


def test_chunking_and_coverage_gates() -> None:
    assert chunking_gate({"documents_considered": 10, "documents_failed": 2})["passed"]
    with pytest.raises(GateFailed):
        chunking_gate({"documents_considered": 10, "documents_failed": 4}, max_failed_ratio=0.3)
    assert coverage_gate({"coverage": 0.995})["passed"]
    with pytest.raises(GateFailed):
        coverage_gate({"coverage": 0.9}, min_coverage=0.99)
    assert _int_or_none(3.0) == 3
    assert _int_or_none("x") is None


def _files(prefix: str, spec: dict[str, list[int]]) -> list[FileInfo]:
    now = datetime.now(UTC)
    out = []
    for partition, sizes in spec.items():
        for i, size in enumerate(sizes):
            key = f"{prefix}{partition + '/' if partition else ''}part-{i}.parquet"
            out.append(FileInfo(key, size, now))
    return out


class FakeIO:
    def __init__(self, listing: dict[tuple[str, str], list[FileInfo]]) -> None:
        self.listing = listing

    def list_files(
        self, layer: Layer, dataset: str, partition: str | None = None
    ) -> list[FileInfo]:
        return self.listing.get((layer.value, dataset), [])


def test_compaction_planning_thresholds() -> None:
    prefix = "silver/chunks/"
    files = _files(
        prefix,
        {
            "source_type=markdown/ingest_date=2026-09-01": [1_000] * 3,
            "source_type=pdf/ingest_date=2026-09-01": [100 * 1024 * 1024] * 2,
            "source_type=html/ingest_date=2026-09-01": [50 * 1024 * 1024] * 9,
        },
    )
    groups = group_by_partition(prefix, files)
    assert set(groups) == {
        "source_type=markdown/ingest_date=2026-09-01",
        "source_type=pdf/ingest_date=2026-09-01",
        "source_type=html/ingest_date=2026-09-01",
    }
    io = FakeIO({("silver", "chunks"): files})
    plans = plan_partitions(io, min_files=8, small_file_mb=32)  # type: ignore[arg-type]
    by_partition = {p.partition: p for p in plans}
    assert "source_type=markdown/ingest_date=2026-09-01" in by_partition  # small files
    assert "source_type=html/ingest_date=2026-09-01" in by_partition  # too many files
    assert (
        "source_type=pdf/ingest_date=2026-09-01" not in by_partition
    )  # 2 large files: leave alone
    assert by_partition["source_type=markdown/ingest_date=2026-09-01"].bytes_before == 3_000
    assert all(
        layer != Layer.BRONZE or ds != "raw" for layer, ds, _ in COMPACTABLE
    )  # bronze/raw is immutable
    assert compaction_layers() == ["bronze", "silver", "gold"]


def test_dag_configs_load_and_validate() -> None:
    sys.path.insert(0, str(ROOT / "airflow" / "plugins"))
    import os

    os.environ["AKL_DAG_CONFIG_DIR"] = str(ROOT / "configs" / "dags")
    from importlib import reload

    from akl_airflow import common

    reload(common)
    names = ["ingestion", "chunking", "embedding", "qdrant_sync", "maintenance"]
    cfgs = {n: common.dag_config(n) for n in names}
    assert [c.dag_id for c in cfgs.values()] == [f"akl_{n}" for n in names]
    assert cfgs["ingestion"].schedule == "*/30 * * * *"
    assert cfgs["ingestion"].gates["max_quarantine_ratio"] == 0.25
    assert cfgs["embedding"].timeout("embed_backlog").total_seconds() == 5400
    assert cfgs["embedding"].get("pool") == "akl_embedding"
    assert cfgs["maintenance"].get("retention_days")["bronze"] == 365
    args = common.default_args(cfgs["qdrant_sync"])
    assert args["retries"] == 3
    assert args["retry_exponential_backoff"] is True
    assert args["sla"].total_seconds() == 1800
    missing = common.dag_config("does_not_exist")
    assert missing.dag_id == "akl_does_not_exist"
    assert missing.schedule is None
