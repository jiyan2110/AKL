"""Unit tests for Gold projection, views, and coverage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pyarrow as pa
import pytest

from akl.config import Settings
from akl.lakehouse.engine import DuckDBEngine
from akl.lakehouse.gold import build_retrieval_units
from akl.lakehouse.io import Layer
from akl.lakehouse.schemas import DatasetSchema, enforce
from akl.lakehouse.schemas.gold import CHUNK_EMBEDDINGS, DEFAULT_EMBEDDING_VERSION, RETRIEVAL_UNITS
from akl.lakehouse.schemas.silver import CHUNKS, DOCUMENTS
from akl.lakehouse.silver import SilverStore
from akl.lakehouse.views import ViewRegistry

pytestmark = pytest.mark.unit
T0 = datetime(2026, 9, 4, 10, 0, tzinfo=UTC)
EV = DEFAULT_EMBEDDING_VERSION


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> DuckDBEngine:
    for key in ("AKL_DB_PASSWORD", "AKL_S3_ACCESS_KEY", "AKL_S3_SECRET_KEY"):
        monkeypatch.setenv(key, "x")
    monkeypatch.setenv("AKL_DUCKDB_THREADS", "1")
    monkeypatch.setenv("AKL_DUCKDB_MEMORY_LIMIT", "256MB")
    return DuckDBEngine(Settings.load(config_file=None, env_file=None))


def doc(doc_id: str, version: str, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "document_version_id": version,
        "document_id": doc_id,
        "content_sha256": "c" * 64,
        "source_type": "github",
        "canonical_source_uri": f"github://org/repo/main/{doc_id}.md",
        "title": f"Title {doc_id}",
        "language": "en",
        "text": "hello world",
        "quality_score": 0.9,
        "security_level": "internal",
        "parser_name": "test",
        "parser_version": "1.0.0",
        "parsed_at": T0,
        "metadata": {"git.repo": "org/repo", "git.branch": "main", "git.path": f"{doc_id}.md"},
    }
    base.update(over)
    return SilverStore.prepare_document_row(base)


def chunk(chunk_id: str, doc_id: str, version: str, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "chunk_id": chunk_id,
        "chunk_key": "k" * 40,
        "lineage_id": chunk_id,
        "chunk_checksum": "d" * 64,
        "embedded_text_sha256": "e" * 64,
        "document_id": doc_id,
        "document_version_id": version,
        "chunk_index": 0,
        "chunk_type": "prose",
        "heading_path": ["Install", "Docker"],
        "text": "hello",
        "start_char": 0,
        "end_char": 5,
        "token_count": 2,
        "overlap_prev_tokens": 0,
        "quality_score": 0.8,
        "quality_flags": [],
        "chunker_version": "1.0.0",
        "chunk_config_hash": "abcd1234abcd1234",
        "security_level": "internal",
        "source_type": "github",
        "created_at": T0,
        "is_current": True,
        "is_deleted": False,
        "ingest_date": T0.date(),
    }
    base.update(over)
    return base


class Mem:
    def __init__(self, engine: DuckDBEngine) -> None:
        self.engine = engine
        self.tables: dict[str, pa.Table] = {}

    def set(self, dataset: str, schema: DatasetSchema, rows: list[dict[str, Any]]) -> None:
        table = enforce(pa.Table.from_pylist(rows, schema=schema.schema), schema)
        self.tables[dataset] = table
        self.engine.register(f"t_{dataset.replace('/', '_')}", table)

    def resolve(self, layer: Layer, dataset: str, schema: DatasetSchema) -> str:
        if dataset not in self.tables:
            self.set(dataset, schema, [])
        return f"t_{dataset.replace('/', '_')}"

    def views(self) -> ViewRegistry:
        registry = ViewRegistry(self.engine, resolver=self.resolve)
        registry.register_all()
        return registry


def test_projection_filters_and_extracts_metadata(engine: DuckDBEngine) -> None:
    mem = Mem(engine)
    mem.set(
        "documents",
        DOCUMENTS,
        [
            doc("good", "v1"),
            doc("dup", "v1", is_duplicate_of="good"),
            doc("lowdoc", "v1", quality_score=0.1),
        ],
    )
    mem.set(
        "chunks",
        CHUNKS,
        [
            chunk("c-good", "good", "v1"),
            chunk("c-lowq", "good", "v1", chunk_index=1, quality_score=0.1),
            chunk("c-flag", "good", "v1", chunk_index=2, quality_flags=["low_quality"]),
            chunk("c-dup", "dup", "v1"),
            chunk("c-lowdoc", "lowdoc", "v1"),
        ],
    )
    mem.views()
    rows = build_retrieval_units(engine, mem.resolve, gold_snapshot_id="snap-1").to_pylist()
    assert [row["chunk_id"] for row in rows] == ["c-good"]
    assert rows[0]["repo"] == "org/repo"
    assert rows[0]["heading_breadcrumb"] == "Install › Docker"
    assert rows[0]["gold_snapshot_id"] == "snap-1"
    assert rows[0]["document_updated_at"] is not None
    assert (
        enforce(pa.Table.from_pylist(rows, schema=RETRIEVAL_UNITS.schema), RETRIEVAL_UNITS).num_rows
        == 1
    )


def test_projection_is_incremental_by_chunk_id(engine: DuckDBEngine) -> None:
    mem = Mem(engine)
    mem.set("documents", DOCUMENTS, [doc("d", "v1")])
    mem.set("chunks", CHUNKS, [chunk("c1", "d", "v1"), chunk("c2", "d", "v1", chunk_index=1)])
    mem.views()
    first = build_retrieval_units(engine, mem.resolve, gold_snapshot_id="s1")
    assert first.num_rows == 2
    mem.set("retrieval_units", RETRIEVAL_UNITS, first.to_pylist())
    mem.views()
    assert build_retrieval_units(engine, mem.resolve, gold_snapshot_id="s2").num_rows == 0


def test_active_units_retire_when_silver_changes(engine: DuckDBEngine) -> None:
    mem = Mem(engine)
    mem.set("documents", DOCUMENTS, [doc("d", "v1")])
    mem.set("chunks", CHUNKS, [chunk("c1", "d", "v1")])
    mem.views()
    mem.set(
        "retrieval_units",
        RETRIEVAL_UNITS,
        build_retrieval_units(engine, mem.resolve, gold_snapshot_id="s1").to_pylist(),
    )
    assert mem.views().counts()["v_gold_active_units"] == 1
    mem.set(
        "documents", DOCUMENTS, [doc("d", "v1"), doc("d", "v2", parsed_at=T0 + timedelta(hours=1))]
    )
    mem.set(
        "chunks",
        CHUNKS,
        [chunk("c1", "d", "v1"), chunk("c1b", "d", "v2", created_at=T0 + timedelta(hours=1))],
    )
    assert mem.views().counts()["v_gold_active_units"] == 0


def test_embedding_coverage_backlog_and_stale(engine: DuckDBEngine) -> None:
    mem = Mem(engine)
    mem.set("documents", DOCUMENTS, [doc("d", "v1")])
    mem.set("chunks", CHUNKS, [chunk("c1", "d", "v1"), chunk("c2", "d", "v1", chunk_index=1)])
    mem.views()
    mem.set(
        "retrieval_units",
        RETRIEVAL_UNITS,
        build_retrieval_units(engine, mem.resolve, gold_snapshot_id="s1").to_pylist(),
    )

    def emb(chunk_id: str, sha: str, version: str = EV) -> dict[str, Any]:
        return {
            "chunk_id": chunk_id,
            "chunk_checksum": "d" * 64,
            "embedded_text_sha256": sha,
            "embedding_version": version,
            "model_id": "model",
            "model_version": "1.0",
            "dim": 384,
            "vector": [0.0] * 384,
            "embedded_at": T0,
            "embedder_version": "0.1.0",
            "source_type": "github",
        }

    mem.set(
        "chunk_embeddings",
        CHUNK_EMBEDDINGS,
        [emb("c1", "e" * 64), emb("c2", "0" * 64), emb("c2", "e" * 64, "other")],
    )
    mem.views()
    coverage = {
        row["chunk_id"]: row
        for row in engine.execute(
            "SELECT * FROM v_embedding_coverage ORDER BY chunk_id"
        ).to_pylist()
    }
    assert coverage["c1"]["has_embedding"] is True
    assert coverage["c1"]["stale_embedding"] is False
    assert coverage["c2"]["has_embedding"] is True
    assert coverage["c2"]["stale_embedding"] is True


def test_gold_schema_shapes() -> None:
    assert RETRIEVAL_UNITS.partition_by == ("source_type", "security_level")
    assert CHUNK_EMBEDDINGS.partition_by == ("embedding_version", "source_type")
    assert CHUNK_EMBEDDINGS.schema.field("vector").type == pa.list_(pa.float32())
