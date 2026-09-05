"""Silver stores, current-state views, and append-only tombstones."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa

from akl.lakehouse.engine import QueryEngine
from akl.lakehouse.io import LakehouseIO, Layer, WriteResult
from akl.lakehouse.schemas import DatasetSchema, enforce
from akl.lakehouse.schemas.silver import CHUNKS, DEDUP_LEDGER, DOCUMENTS
from akl.lakehouse.views import ViewRegistry


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sql_in(values: Iterable[str]) -> str:
    return ", ".join("'" + value.replace("'", "''") + "'" for value in values)


class SilverStore:
    def __init__(
        self, io: LakehouseIO, engine: QueryEngine, *, view_params: Mapping[str, str] | None = None
    ) -> None:
        self._io = io
        self._engine = engine
        self._views = ViewRegistry(engine, io, params=view_params)
        self._views_ready = False

    def ensure_views(self, *, refresh: bool = False) -> None:
        if refresh or not self._views_ready:
            self._views.register_all()
            self._views_ready = True

    def view_counts(self) -> dict[str, int]:
        self.ensure_views(refresh=True)
        return self._views.counts()

    def _write(self, table: pa.Table, dataset_schema: DatasetSchema, run_id: str) -> WriteResult:
        dataset = dataset_schema.name.split("/", 1)[1]
        result = self._io.write(
            enforce(table, dataset_schema),
            Layer.SILVER,
            dataset,
            run_id=run_id,
            schema_version=dataset_schema.version,
            partition_by=dataset_schema.partition_by,
            sort_by=dataset_schema.sort_by,
        )
        self._views_ready = False
        return result

    def write_documents(
        self, rows: Sequence[Mapping[str, Any]] | pa.Table, *, run_id: str
    ) -> WriteResult:
        table = (
            rows
            if isinstance(rows, pa.Table)
            else pa.Table.from_pylist(list(rows), schema=DOCUMENTS.schema)
        )
        return self._write(table, DOCUMENTS, run_id)

    def write_chunks(
        self, rows: Sequence[Mapping[str, Any]] | pa.Table, *, run_id: str
    ) -> WriteResult:
        table = (
            rows
            if isinstance(rows, pa.Table)
            else pa.Table.from_pylist(list(rows), schema=CHUNKS.schema)
        )
        return self._write(table, CHUNKS, run_id)

    def write_dedup_ledger(
        self, rows: Sequence[Mapping[str, Any]] | pa.Table, *, run_id: str
    ) -> WriteResult:
        table = (
            rows
            if isinstance(rows, pa.Table)
            else pa.Table.from_pylist(list(rows), schema=DEDUP_LEDGER.schema)
        )
        return self._write(table, DEDUP_LEDGER, run_id)

    def current_documents(
        self, *, where: str | None = None, columns: Sequence[str] | None = None
    ) -> pa.Table:
        self.ensure_views()
        selected = ", ".join(f'"{column}"' for column in columns) if columns else "*"
        sql = f"SELECT {selected} FROM v_current_documents"  # noqa: S608 - columns are caller-selected internal fields
        if where:
            sql += f" WHERE {where}"
        return self._engine.execute(sql)

    def current_chunks(
        self, *, document_id: str | None = None, where: str | None = None
    ) -> pa.Table:
        self.ensure_views()
        clauses: list[str] = []
        if document_id:
            clauses.append(f"document_id = '{document_id}'")
        if where:
            clauses.append(f"({where})")
        sql = "SELECT * FROM v_current_chunks"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return self._engine.execute(sql + " ORDER BY document_id, chunk_index")

    def tombstone_documents(self, document_ids: Sequence[str], *, run_id: str) -> tuple[int, int]:
        if not document_ids:
            return (0, 0)
        self.ensure_views(refresh=True)
        ids_sql = _sql_in(document_ids)
        now = datetime.now(UTC).isoformat()
        docs = self._engine.execute(
            f"""
            SELECT * REPLACE (true AS is_deleted, false AS is_current, TIMESTAMPTZ '{now}' AS parsed_at)
            FROM v_current_documents
            WHERE document_id IN ({ids_sql})
            """  # noqa: S608 - ids and timestamp are escaped/generated above
        )
        chunks = self._engine.execute(
            f"""
            SELECT * REPLACE (true AS is_deleted, false AS is_current, TIMESTAMPTZ '{now}' AS created_at)
            FROM v_current_chunks
            WHERE document_id IN ({ids_sql})
            """  # noqa: S608 - ids and timestamp are escaped/generated above
        )
        if chunks.num_rows:
            self.write_chunks(chunks, run_id=run_id)
        if docs.num_rows:
            self.write_documents(docs, run_id=run_id)
        return (docs.num_rows, chunks.num_rows)

    @staticmethod
    def prepare_document_row(row: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(row)
        text = str(out["text"])
        out.setdefault("text_sha256", text_sha256(text))
        out.setdefault("word_count", len(text.split()))
        out.setdefault("char_count", len(text))
        now = datetime.now(UTC)
        out.setdefault("parsed_at", now)
        out.setdefault("ingest_date", out["parsed_at"].date())
        out.setdefault("is_current", True)
        out.setdefault("is_deleted", False)
        out.setdefault("quality_flags", [])
        out.setdefault("allowed_groups", [])
        if isinstance(out.get("metadata"), dict):
            out["metadata"] = list(out["metadata"].items())
        return out
