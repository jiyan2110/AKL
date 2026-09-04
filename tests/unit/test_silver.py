"""Unit tests for Silver schemas, views, and store helpers."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pyarrow as pa
import pytest

from akl.config import Settings
from akl.lakehouse.engine import DuckDBEngine
from akl.lakehouse.io import Layer
from akl.lakehouse.schemas import DatasetSchema, enforce
from akl.lakehouse.schemas.silver import CHUNKS, DEDUP_LEDGER, DOCUMENTS
from akl.lakehouse.silver import SilverStore
from akl.lakehouse.views import VIEWS, ViewRegistry, render_view_sql

pytestmark = pytest.mark.unit
T0 = datetime(2026, 9, 4, 10, 0, tzinfo=UTC)


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> DuckDBEngine:
    for key in ("AKL_DB_PASSWORD", "AKL_S3_ACCESS_KEY", "AKL_S3_SECRET_KEY"):
        monkeypatch.setenv(key, "x")
    monkeypatch.setenv("AKL_DUCKDB_THREADS", "1")
    monkeypatch.setenv("AKL_DUCKDB_MEMORY_LIMIT", "256MB")
    return DuckDBEngine(Settings.load(config_file=None, env_file=None))


def doc_row(doc: str, version: str, parsed_at: datetime, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "document_version_id": version,
        "document_id": doc,
        "content_sha256": "c" * 64,
        "source_type": "markdown",
        "canonical_source_uri": f"https://example.com/{doc}",
        "title": "T",
        "language": "en",
        "text": "hello world",
        "quality_score": 0.9,
        "security_level": "internal",
        "parser_name": "test",
        "parser_version": "1.0.0",
        "parsed_at": parsed_at,
    }
    base.update(over)
    return SilverStore.prepare_document_row(base)


def chunk_row(
    chunk: str, doc: str, version: str, created_at: datetime, **over: Any
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "chunk_id": chunk,
        "chunk_key": "k" * 40,
        "lineage_id": chunk,
        "chunk_checksum": "d" * 64,
        "document_id": doc,
        "document_version_id": version,
        "chunk_index": 0,
        "chunk_type": "prose",
        "text": "hello",
        "start_char": 0,
        "end_char": 5,
        "token_count": 2,
        "overlap_prev_tokens": 0,
        "quality_score": 0.8,
        "chunker_version": "1.0.0",
        "chunk_config_hash": "abcd1234abcd1234",
        "security_level": "internal",
        "source_type": "markdown",
        "created_at": created_at,
        "is_current": True,
        "is_deleted": False,
        "ingest_date": created_at.date(),
    }
    base.update(over)
    return base


def test_schema_shapes() -> None:
    assert len(DOCUMENTS.columns) == 30
    assert len(CHUNKS.columns) == 37
    assert DEDUP_LEDGER.partition_by == ()
    assert DOCUMENTS.partition_by == ("source_type", "ingest_date")
    assert not DOCUMENTS.schema.field("text").nullable
    assert CHUNKS.schema.field("embedded_text_sha256").nullable


def test_prepare_document_row_fills_derived_fields() -> None:
    row = doc_row("d1", "v1", T0, metadata={"k": "v"})
    assert row["word_count"] == 2
    assert row["char_count"] == 11
    assert len(row["text_sha256"]) == 64
    assert row["ingest_date"] == date(2026, 9, 4)
    assert row["metadata"] == [("k", "v")]
    assert enforce(pa.Table.from_pylist([row], schema=DOCUMENTS.schema), DOCUMENTS).num_rows == 1


def test_render_view_sql_substitutes_sources() -> None:
    def resolver(layer: Layer, dataset: str, schema: DatasetSchema) -> str:
        return f"src_{dataset}"

    sql = render_view_sql(VIEWS[0], resolver)
    assert "src_documents" in sql
    assert "{{" not in sql


def test_empty_current_views_have_zero_counts(engine: DuckDBEngine) -> None:
    registry = _register(engine, [], [])
    assert registry.counts() == {"v_current_documents": 0, "v_current_chunks": 0}


def _register(
    engine: DuckDBEngine, docs: list[dict[str, Any]], chunks: list[dict[str, Any]]
) -> ViewRegistry:
    engine.register(
        "t_docs", enforce(pa.Table.from_pylist(docs, schema=DOCUMENTS.schema), DOCUMENTS)
    )
    engine.register("t_chunks", enforce(pa.Table.from_pylist(chunks, schema=CHUNKS.schema), CHUNKS))

    def resolver(layer: Layer, dataset: str, schema: DatasetSchema) -> str:
        return {"documents": "t_docs", "chunks": "t_chunks"}[dataset]

    registry = ViewRegistry(engine, resolver=resolver)
    registry.register_all()
    return registry


def test_current_documents_latest_version_wins(engine: DuckDBEngine) -> None:
    _register(
        engine,
        [
            doc_row("d1", "v1", T0),
            doc_row("d1", "v2", T0 + timedelta(hours=1)),
            doc_row("d2", "v9", T0),
        ],
        [],
    )
    output = engine.execute(
        "SELECT document_id, document_version_id FROM v_current_documents ORDER BY 1"
    ).to_pylist()
    assert output == [
        {"document_id": "d1", "document_version_id": "v2"},
        {"document_id": "d2", "document_version_id": "v9"},
    ]


def test_tombstone_row_removes_document_and_chunks(engine: DuckDBEngine) -> None:
    docs = [
        doc_row("d1", "v1", T0),
        doc_row("d1", "v1", T0 + timedelta(hours=2), is_deleted=True, is_current=False),
    ]
    registry = _register(engine, docs, [chunk_row("c1", "d1", "v1", T0)])
    assert registry.counts() == {"v_current_documents": 0, "v_current_chunks": 0}


def test_current_chunks_follow_current_version(engine: DuckDBEngine) -> None:
    docs = [doc_row("d1", "v1", T0), doc_row("d1", "v2", T0 + timedelta(hours=1))]
    chunks = [
        chunk_row("c-old", "d1", "v1", T0),
        chunk_row("c-new", "d1", "v2", T0 + timedelta(hours=1)),
        chunk_row("c-new", "d1", "v2", T0 + timedelta(hours=2), is_current=False),
    ]
    registry = _register(engine, docs, chunks)
    assert engine.execute("SELECT chunk_id FROM v_current_chunks").to_pylist() == []
    assert registry.counts()["v_current_documents"] == 1


def test_current_chunks_positive(engine: DuckDBEngine) -> None:
    _register(
        engine,
        [doc_row("d1", "v2", T0)],
        [chunk_row("c1", "d1", "v2", T0), chunk_row("c2", "d1", "v2", T0, chunk_index=1)],
    )
    ids = engine.execute("SELECT chunk_id FROM v_current_chunks ORDER BY chunk_index").to_pylist()
    assert [row["chunk_id"] for row in ids] == ["c1", "c2"]
