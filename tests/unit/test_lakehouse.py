"""Unit tests for the lakehouse layer without MinIO."""

from __future__ import annotations

import pyarrow as pa
import pytest

from akl.config import Settings
from akl.lakehouse.engine import DuckDBEngine
from akl.lakehouse.io import (
    LakehouseIO,
    LakehouseIOError,
    Layer,
    dataset_prefix,
    dataset_uri,
    partition_path,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    for key in ("AKL_DB_PASSWORD", "AKL_S3_ACCESS_KEY", "AKL_S3_SECRET_KEY"):
        monkeypatch.setenv(key, "x")
    monkeypatch.setenv("AKL_S3_BUCKET", "unit-bucket")
    monkeypatch.setenv("AKL_DUCKDB_THREADS", "1")
    monkeypatch.setenv("AKL_DUCKDB_MEMORY_LIMIT", "256MB")
    return Settings.load(config_file=None, env_file=None)


def test_dataset_uri_and_prefix() -> None:
    assert dataset_uri("b", Layer.SILVER, "chunks") == "s3://b/silver/chunks"
    assert dataset_uri("b", "gold", "/retrieval_units/") == "s3://b/gold/retrieval_units"
    assert dataset_prefix(Layer.BRONZE, "manifest") == "bronze/manifest/"


def test_partition_path_preserves_order() -> None:
    assert (
        partition_path(source_type="pdf", ingest_date="2026-09-04")
        == "source_type=pdf/ingest_date=2026-09-04"
    )


def test_engine_executes_locally_without_s3(settings: Settings) -> None:
    with DuckDBEngine(settings) as engine:
        assert engine.execute_scalar("SELECT 40 + 2") == 42
        assert engine.execute("SELECT 1 AS a UNION ALL SELECT 2").num_rows == 2


def test_engine_registers_arrow_table(settings: Settings) -> None:
    with DuckDBEngine(settings) as engine:
        engine.register("t", pa.table({"x": [3, 1, 2]}))
        assert engine.execute_scalar("SELECT sum(x) FROM t") == 6
        engine.unregister("t")


def test_read_expression_glob(settings: Settings) -> None:
    with DuckDBEngine(settings) as engine:
        io = LakehouseIO(settings, engine)
        assert (
            io.read_expression(Layer.SILVER, "chunks")
            == "read_parquet('s3://unit-bucket/silver/chunks/**/*.parquet', hive_partitioning = true, union_by_name = true)"
        )
        assert "silver/chunks/source_type=pdf/*.parquet" in io.read_expression(
            Layer.SILVER, "chunks", partition="source_type=pdf"
        )


def test_write_empty_table_is_noop(settings: Settings) -> None:
    with DuckDBEngine(settings) as engine:
        result = LakehouseIO(settings, engine).write(
            pa.table({"a": pa.array([], pa.int64())}),
            Layer.SCRATCH,
            "x",
            run_id="r",
            schema_version="1",
        )
        assert result.rows == 0
        assert result.files == ()


def test_write_rejects_unknown_partition_column(settings: Settings) -> None:
    with DuckDBEngine(settings) as engine:
        io = LakehouseIO(settings, engine)
        with pytest.raises(LakehouseIOError):
            io.write(
                pa.table({"a": [1]}),
                Layer.SCRATCH,
                "x",
                run_id="r",
                schema_version="1",
                partition_by=["nope"],
            )


def test_delete_refuses_bronze_raw(settings: Settings) -> None:
    with DuckDBEngine(settings) as engine:
        with pytest.raises(LakehouseIOError, match="immutable"):
            LakehouseIO(settings, engine).delete_keys(["bronze/raw/source_type=pdf/sha256=abc.pdf"])
