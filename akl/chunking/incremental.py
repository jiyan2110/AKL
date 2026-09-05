"""Incremental chunk update (PRD §4.12) and the ChunkingService (PRD §7.4 task logic)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

import pyarrow as pa

from akl.chunking.engine import ChunkingEngine
from akl.chunking.models import Chunk, ChunkDiff, ChunkStatus
from akl.chunking.semantic import Embedder
from akl.chunking.tokenizer import TokenCounter
from akl.config import Settings
from akl.db.models import Document, DocumentVersion
from akl.db.repositories.chunks import ChunkRepository
from akl.db.session import Database
from akl.errors import AKLError
from akl.ingestion.models import (
    FetchedObject,
    SourceItem,
    UnifiedDocument,
    document_from_silver_row,
)
from akl.ingestion.parsers.base import BaseParser
from akl.ingestion.registry import ParserRegistry
from akl.lakehouse.bronze import BronzeStore
from akl.lakehouse.engine import DuckDBEngine
from akl.lakehouse.gold import GoldStore
from akl.lakehouse.io import LakehouseIO, Layer
from akl.lakehouse.silver import SilverStore


class ExistingChunk(Protocol):
    """Minimal shape of a current chunk record (ORM row or test stub)."""

    @property
    def chunk_id(self) -> uuid.UUID: ...

    @property
    def chunk_key(self) -> str: ...

    @property
    def chunk_checksum(self) -> str: ...

    @property
    def lineage_id(self) -> uuid.UUID: ...


def diff_chunks(
    document_id: uuid.UUID, new_chunks: Sequence[Chunk], existing: Sequence[ExistingChunk]
) -> ChunkDiff:
    """Classify new chunks against the current set (PRD §4.12): unchanged / modified / moved / added / removed."""
    diff = ChunkDiff(document_id=document_id)
    old_by_key = {c.chunk_key: c for c in existing}
    old_by_sum: dict[str, list[ExistingChunk]] = {}
    for c in existing:
        old_by_sum.setdefault(c.chunk_checksum, []).append(c)
    new_ids: set[uuid.UUID] = set()
    claimed: set[uuid.UUID] = set()

    for chunk in new_chunks:
        new_ids.add(chunk.chunk_id)
        old = old_by_key.get(chunk.chunk_key)
        if old is not None and old.chunk_checksum == chunk.chunk_checksum:
            chunk.status = ChunkStatus.UNCHANGED
            chunk.lineage_id = old.lineage_id
            chunk.chunk_id = old.chunk_id
            claimed.add(old.chunk_id)
            diff.unchanged += 1
            continue
        if old is not None:
            chunk.status = ChunkStatus.MODIFIED
            chunk.lineage_id = old.lineage_id
            claimed.add(old.chunk_id)
            diff.modified += 1
        else:
            moved = next(
                (c for c in old_by_sum.get(chunk.chunk_checksum, []) if c.chunk_id not in claimed),
                None,
            )
            if moved is not None:
                chunk.status = ChunkStatus.MOVED
                chunk.lineage_id = moved.lineage_id
                claimed.add(moved.chunk_id)
                diff.moved += 1
            else:
                chunk.status = ChunkStatus.ADDED
                chunk.lineage_id = chunk.chunk_id
                diff.added += 1
        diff.to_write.append(chunk)

    for old_chunk in existing:
        if old_chunk.chunk_id in new_ids:
            continue
        if old_chunk.chunk_id in claimed:
            diff.superseded_ids.append(old_chunk.chunk_id)  # replaced by a modified/moved chunk
        else:
            diff.removed_ids.append(old_chunk.chunk_id)
            diff.removed += 1
    return diff


@dataclass
class ChunkRunReport:
    run_id: str
    documents_considered: int = 0
    documents_chunked: int = 0
    documents_unchanged: int = 0
    documents_failed: int = 0
    chunks_written: int = 0
    chunks_tombstoned: int = 0
    unchanged: int = 0
    modified: int = 0
    moved: int = 0
    added: int = 0
    removed: int = 0
    reparented: int = 0
    gold_rows_promoted: int = 0
    gold_snapshot_id: str | None = None
    failures: list[dict[str, str]] = field(default_factory=list)


class ChunkingService:
    """Chunk current documents that lack chunks for the active chunker version/config, then refresh Gold."""

    def __init__(
        self,
        settings: Settings,
        engine: DuckDBEngine,
        db: Database,
        *,
        parsers: ParserRegistry | None = None,
        embedder: Embedder | None = None,
        counter: TokenCounter | None = None,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.db = db
        self.io = LakehouseIO(settings, engine)
        self.bronze = BronzeStore(self.io)
        self.chunker = ChunkingEngine(
            settings.chunking,
            counter or TokenCounter(models_dir=settings.core.models_dir),
            embedder,
        )
        self.view_params = {
            "chunker_version": settings.chunking.chunker_version,
            "chunk_config_hash": self.chunker.config_hash,
        }
        self.silver = SilverStore(self.io, engine, view_params=self.view_params)
        self.gold = GoldStore(self.io, engine, view_params=self.view_params)
        if parsers is None:
            from akl.ingestion.registry import ConnectorRegistry
            from akl.ingestion.service import register_builtins

            parsers = ParserRegistry()
            register_builtins(ConnectorRegistry(), parsers)
        self.parsers = parsers

    # -- public ------------------------------------------------------------------------------
    def run(
        self,
        *,
        run_id: str,
        document_ids: Sequence[uuid.UUID] | None = None,
        limit: int = 200,
        refresh_gold: bool = True,
    ) -> ChunkRunReport:
        report = ChunkRunReport(run_id=run_id)
        now = datetime.now(UTC)
        new_rows: list[dict[str, Any]] = []
        tombstones: list[pa.Table] = []
        with self.db.session() as session:
            repo = ChunkRepository(session)
            todo = repo.documents_needing_chunks(
                chunker_version=self.settings.chunking.chunker_version,
                chunk_config_hash=self.chunker.config_hash,
                limit=limit,
                document_ids=document_ids,
            )
            for doc_row, version in todo:
                report.documents_considered += 1
                try:
                    unified = self._load_document(doc_row, version)
                    chunks = self.chunker.chunk(unified)
                except AKLError as exc:
                    report.documents_failed += 1
                    report.failures.append(
                        {
                            "document_id": str(doc_row.document_id),
                            "code": exc.code,
                            "error": exc.message,
                        }
                    )
                    continue
                except (KeyError, ValueError) as exc:
                    report.documents_failed += 1
                    report.failures.append(
                        {
                            "document_id": str(doc_row.document_id),
                            "code": "AKL-E4001",
                            "error": str(exc),
                        }
                    )
                    continue
                existing = repo.current_for_document(doc_row.document_id)
                diff = diff_chunks(doc_row.document_id, chunks, existing)
                report.unchanged += diff.unchanged
                report.modified += diff.modified
                report.moved += diff.moved
                report.added += diff.added
                report.removed += diff.removed
                if not diff.changed:
                    report.documents_unchanged += 1
                # neighbours must reference final ids (unchanged chunks keep their old ids)
                for prev, cur in zip(chunks, chunks[1:], strict=False):
                    prev.next_chunk_id = cur.chunk_id
                    cur.prev_chunk_id = prev.chunk_id
                for chunk in diff.to_write:
                    new_rows.append(chunk.to_silver_row(created_at=now))
                # A new document version re-parents unchanged chunks: their Silver rows must carry the new
                # document_version_id for v_current_chunks (same chunk_id/checksum → no re-embedding).
                old_versions = {c.chunk_id: c.document_version_id for c in existing}
                for chunk in chunks:
                    if (
                        chunk.status is ChunkStatus.UNCHANGED
                        and old_versions.get(chunk.chunk_id) != chunk.document_version_id
                    ):
                        new_rows.append(chunk.to_silver_row(created_at=now))
                        report.reparented += 1
                if diff.superseded_ids or diff.removed_ids:
                    tombstones.append(
                        self._tombstone_rows(
                            doc_row.document_id, [*diff.superseded_ids, *diff.removed_ids], now
                        )
                    )
                # Postgres current-state index
                repo.upsert_current(
                    [self._pg_row(c) for c in chunks if c.status is not ChunkStatus.UNCHANGED]
                )
                # an UNCHANGED chunk from an older chunker version must be re-stamped so the backlog query clears
                repo.upsert_current(
                    [self._pg_row(c) for c in chunks if c.status is ChunkStatus.UNCHANGED]
                )
                repo.retire(diff.superseded_ids, deleted=False)
                repo.retire(diff.removed_ids, deleted=True)
                report.documents_chunked += 1

            if tombstones:
                combined = pa.concat_tables(tombstones)
                self.silver.write_chunks(combined, run_id=run_id)
                report.chunks_tombstoned = combined.num_rows
            if new_rows:
                self.silver.write_chunks(new_rows, run_id=run_id)
                report.chunks_written = len(new_rows)

        if refresh_gold and (new_rows or tombstones or report.documents_chunked):
            result, snapshot = self.gold.refresh_retrieval_units(
                run_id=run_id,
                chunk_quality_min=self.settings.chunking.chunk_quality_min,
                doc_quality_min=self.settings.chunking.doc_quality_min,
            )
            report.gold_rows_promoted = result.rows
            report.gold_snapshot_id = snapshot
        return report

    # -- helpers -------------------------------------------------------------------------------
    def _load_document(self, doc_row: Document, version: DocumentVersion) -> UnifiedDocument:
        table = self.silver.current_documents(
            where=f"document_version_id = '{version.document_version_id}'"
        )
        if table.num_rows == 0:
            raise AKLError(
                f"silver row missing for version {version.document_version_id}",
                details={"document_id": str(doc_row.document_id)},
            )
        row = table.to_pylist()[0]
        try:
            return document_from_silver_row(row)
        except KeyError:
            return self._reparse(doc_row, version, row)

    def _reparse(
        self, doc_row: Document, version: DocumentVersion, row: dict[str, Any]
    ) -> UnifiedDocument:
        """Rows written before schema 1.1.0 carry no blocks: re-parse the immutable Bronze bytes."""
        ext = version.bronze_object_key.rsplit(".", 1)[-1]
        parser: BaseParser = self.parsers.select(None, ext, doc_row.source_type)
        item = SourceItem(
            uri=doc_row.canonical_source_uri,
            canonical_uri=doc_row.canonical_source_uri,
            source_type=doc_row.source_type,
            filename=f"document.{ext}",
            security_level=doc_row.security_level,
            allowed_groups=tuple(doc_row.allowed_groups),
        )
        data = self.bronze.get_raw(version.bronze_object_key)
        unified = parser.parse(
            FetchedObject.from_bytes(item, data, mime_type=None, fetched_at=version.fetched_at)
        )
        return unified.model_copy(
            update={"language": row.get("language"), "title": row.get("title") or unified.title}
        )

    def _tombstone_rows(
        self, document_id: uuid.UUID, chunk_ids: Sequence[uuid.UUID], now: datetime
    ) -> pa.Table:
        """Tombstone copies of the latest Silver row per chunk_id (PRD §9.11).

        Reads the chunks dataset directly rather than ``v_current_chunks``: by the time
        chunking runs, ``v_current_documents`` already points at the NEW document version,
        so superseded chunks (old version id) are no longer visible through the view.
        """
        ids_sql = ", ".join(f"'{c}'" for c in chunk_ids)
        source = self.io.read_expression(Layer.SILVER, "chunks")
        return self.engine.execute(
            f"""
            WITH latest AS (
                SELECT c.*
                FROM {source} AS c
                WHERE c.document_id = '{document_id}' AND c.chunk_id IN ({ids_sql})
                QUALIFY row_number() OVER (PARTITION BY c.chunk_id ORDER BY c.created_at DESC) = 1
            )
            SELECT * REPLACE (
                false AS is_current,
                true AS is_deleted,
                TIMESTAMPTZ '{now.isoformat()}' AS created_at
            )
            FROM latest
            WHERE NOT is_deleted
            """  # noqa: S608 - uuids only
        )

    @staticmethod
    def _pg_row(chunk: Chunk) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "chunk_key": chunk.chunk_key,
            "lineage_id": chunk.lineage_id,
            "chunk_checksum": chunk.chunk_checksum,
            "embedded_text_sha256": chunk.embedded_text_sha256,
            "document_id": chunk.document_id,
            "document_version_id": chunk.document_version_id,
            "chunk_index": chunk.chunk_index,
            "chunk_type": chunk.chunk_type,
            "token_count": chunk.token_count,
            "quality_score": chunk.quality_score,
            "security_level": chunk.security_level,
            "allowed_groups": list(chunk.allowed_groups),
            "chunker_version": chunk.chunker_version,
            "chunk_config_hash": chunk.chunk_config_hash,
            "is_current": True,
            "is_deleted": False,
        }
