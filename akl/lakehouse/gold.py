"""Gold retrieval-unit projection, embeddings, coverage, and statistics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from akl.errors import AKLError
from akl.lakehouse.engine import QueryEngine
from akl.lakehouse.io import LakehouseIO, Layer, WriteResult
from akl.lakehouse.schemas import DatasetSchema, enforce
from akl.lakehouse.schemas.gold import (
    CHUNK_EMBEDDINGS,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_VERSION,
    EVAL_QA_PAIRS,
    RETRIEVAL_UNITS,
    STATS,
)
from akl.lakehouse.views import SQL_DIR, SourceResolver, ViewRegistry, render_sql

DEFAULT_CHUNK_QUALITY_MIN = 0.30
DEFAULT_DOC_QUALITY_MIN = 0.35
RETRIEVAL_UNITS_SQL = SQL_DIR / "gold" / "retrieval_units.sql"


class GoldError(AKLError):
    code = "AKL-E2002"
    retryable = False


def build_retrieval_units(
    engine: QueryEngine,
    resolver: SourceResolver,
    *,
    gold_snapshot_id: str,
    chunk_quality_min: float = DEFAULT_CHUNK_QUALITY_MIN,
    doc_quality_min: float = DEFAULT_DOC_QUALITY_MIN,
) -> pa.Table:
    sql = render_sql(
        RETRIEVAL_UNITS_SQL.read_text(encoding="utf-8"),
        sources={"retrieval_units": (Layer.GOLD, "retrieval_units", RETRIEVAL_UNITS)},
        resolver=resolver,
        params={
            "gold_snapshot_id": gold_snapshot_id,
            "chunk_quality_min": repr(float(chunk_quality_min)),
            "doc_quality_min": repr(float(doc_quality_min)),
        },
        context="gold/retrieval_units.sql",
    )
    return enforce(engine.execute(sql), RETRIEVAL_UNITS)


class GoldStore:
    def __init__(
        self,
        io: LakehouseIO,
        engine: QueryEngine,
        *,
        embedding_version: str = DEFAULT_EMBEDDING_VERSION,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        view_params: Mapping[str, str] | None = None,
    ) -> None:
        self._io = io
        self._engine = engine
        self.embedding_version = embedding_version
        self.embedding_dim = embedding_dim
        self._views = ViewRegistry(
            engine, io, params={**(view_params or {}), "embedding_version": embedding_version}
        )
        self._views_ready = False

    def ensure_views(self, *, refresh: bool = False) -> None:
        if refresh or not self._views_ready:
            self._views.register_all()
            self._views_ready = True

    def dataset_source(self, dataset: str, schema: DatasetSchema) -> str:
        """SQL expression backing a Gold dataset (glob or empty table) for ad-hoc joins."""
        return self._views.resolver(Layer.GOLD, dataset, schema)

    def view_counts(self) -> dict[str, int]:
        self.ensure_views(refresh=True)
        return self._views.counts()

    def _write(self, table: pa.Table, dataset_schema: DatasetSchema, run_id: str) -> WriteResult:
        dataset = dataset_schema.name.split("/", 1)[1]
        result = self._io.write(
            enforce(table, dataset_schema),
            Layer.GOLD,
            dataset,
            run_id=run_id,
            schema_version=dataset_schema.version,
            partition_by=dataset_schema.partition_by,
            sort_by=dataset_schema.sort_by,
        )
        self._views_ready = False
        return result

    def refresh_retrieval_units(
        self,
        *,
        run_id: str,
        chunk_quality_min: float = DEFAULT_CHUNK_QUALITY_MIN,
        doc_quality_min: float = DEFAULT_DOC_QUALITY_MIN,
    ) -> tuple[WriteResult, str]:
        self.ensure_views(refresh=True)
        table = build_retrieval_units(
            self._engine,
            self._views.resolver,
            gold_snapshot_id=run_id,
            chunk_quality_min=chunk_quality_min,
            doc_quality_min=doc_quality_min,
        )
        if table.num_rows == 0:
            return WriteResult(self._io.uri(Layer.GOLD, "retrieval_units"), 0, (), 0), run_id
        return self._write(table, RETRIEVAL_UNITS, run_id), run_id

    def write_embeddings(
        self, rows: Sequence[Mapping[str, Any]] | pa.Table, *, run_id: str
    ) -> WriteResult:
        table = (
            rows
            if isinstance(rows, pa.Table)
            else pa.Table.from_pylist(list(rows), schema=CHUNK_EMBEDDINGS.schema)
        )
        if table.num_rows:
            lengths = pc.list_value_length(table.column("vector")).to_pylist()
            bad = [index for index, length in enumerate(lengths) if length != self.embedding_dim]
            if bad:
                raise GoldError(
                    f"vector dimension mismatch (expected {self.embedding_dim})",
                    details={"rows": bad[:10], "count": len(bad)},
                )
        return self._write(table, CHUNK_EMBEDDINGS, run_id)

    def write_stats(self, rows: Sequence[Mapping[str, Any]], *, run_id: str) -> WriteResult:
        return self._write(pa.Table.from_pylist(list(rows), schema=STATS.schema), STATS, run_id)

    def write_qa_pairs(self, rows: Sequence[Mapping[str, Any]], *, run_id: str) -> WriteResult:
        return self._write(
            pa.Table.from_pylist(list(rows), schema=EVAL_QA_PAIRS.schema), EVAL_QA_PAIRS, run_id
        )

    def read_qa_pairs(self, *, version: str | None = None) -> pa.Table:
        self.ensure_views(refresh=False)
        source = self.dataset_source("eval/qa_pairs", EVAL_QA_PAIRS)
        where = f"WHERE version = '{version}'" if version else ""
        return self._engine.execute(f"SELECT * FROM {source} {where} ORDER BY qa_id")  # noqa: S608 - version from settings/CLI

    def latest_qa_version(self) -> str | None:
        """Lexicographically greatest version string (run ids embed a timestamp, so this is
        "most recent" in practice — the schema has no separate created_at column)."""
        source = self.dataset_source("eval/qa_pairs", EVAL_QA_PAIRS)
        table = self._engine.execute(
            f"SELECT DISTINCT version FROM {source} ORDER BY version DESC LIMIT 1"  # noqa: S608 - internal column only
        )
        if table.num_rows == 0:
            return None
        return str(table.column("version")[0].as_py())

    def active_units(
        self, *, where: str | None = None, columns: Sequence[str] | None = None
    ) -> pa.Table:
        self.ensure_views()
        selected = ", ".join(f'"{column}"' for column in columns) if columns else "*"
        sql = f"SELECT {selected} FROM v_gold_active_units"  # noqa: S608 - internal column selection
        if where:
            sql += f" WHERE {where}"
        return self._engine.execute(sql + " ORDER BY document_id, chunk_index")

    def embedding_backlog(self) -> pa.Table:
        self.ensure_views()
        return self._engine.execute(
            "SELECT chunk_id, document_id, source_type, security_level, embedded_text_sha256 FROM v_embedding_coverage WHERE NOT has_embedding OR stale_embedding ORDER BY chunk_id"
        )

    def coverage_ratio(self) -> float:
        self.ensure_views()
        total = int(self._engine.execute_scalar("SELECT count(*) FROM v_embedding_coverage") or 0)
        if total == 0:
            return 1.0
        covered = int(
            self._engine.execute_scalar(
                "SELECT count(*) FROM v_embedding_coverage WHERE has_embedding AND NOT stale_embedding"
            )
            or 0
        )
        return covered / total

    def compute_stats(self, *, run_id: str) -> list[dict[str, Any]]:
        self.ensure_views(refresh=True)
        now = datetime.now(UTC)
        rows: list[dict[str, Any]] = []

        def add(metric: str, dimension: str, value: float) -> None:
            rows.append(
                {
                    "metric": metric,
                    "dimension": dimension,
                    "value": float(value),
                    "gold_snapshot_id": run_id,
                    "computed_at": now,
                    "snapshot_date": now.date(),
                }
            )

        for name in ("v_current_documents", "v_current_chunks", "v_gold_active_units"):
            add(
                f"{name}_count",
                "_all",
                int(self._engine.execute_scalar(f"SELECT count(*) FROM {name}") or 0),  # noqa: S608 - view names are registry constants
            )
        add("embedding_coverage_ratio", self.embedding_version, self.coverage_ratio())
        add("embedding_backlog", self.embedding_version, self.embedding_backlog().num_rows)
        if rows:
            self.write_stats(rows, run_id=run_id)
        return rows
