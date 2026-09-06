"""Lineage (PRD §2.9, §9.9): dataset-level Bronze→Silver→Gold edges, and a per-document trace.

``LineageRepository`` writes/reads ``lineage_edges`` rows — one row per pipeline stage per run,
recording which output dataset/partition a task produced from which input, with row counts.
``document_trace()`` is not stored; it is assembled live from Postgres + Gold on request, since a
document's own version/chunk/embedding chain is already fully queryable and duplicating it into
lineage_edges would just be another place for it to go stale.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select

from akl.db.models import Chunk, Document, DocumentVersion, LineageEdge
from akl.db.repositories import Repository


class LineageRepository(Repository):
    def record(
        self,
        *,
        run_id: str,
        task_id: str | None,
        output_dataset: str,
        rows_out: int,
        input_dataset: str | None = None,
        input_partition: str | None = None,
        output_partition: str | None = None,
        rows_in: int | None = None,
    ) -> None:
        self.session.add(
            LineageEdge(
                run_id=run_id,
                task_id=task_id,
                input_dataset=input_dataset,
                input_partition=input_partition,
                output_dataset=output_dataset,
                output_partition=output_partition,
                rows_in=rows_in,
                rows_out=rows_out,
            )
        )

    def for_run(self, run_id: str) -> list[LineageEdge]:
        return list(
            self.session.scalars(
                select(LineageEdge).where(LineageEdge.run_id == run_id).order_by(LineageEdge.id)
            )
        )

    def for_dataset(self, output_dataset: str, *, limit: int = 50) -> list[LineageEdge]:
        return list(
            self.session.scalars(
                select(LineageEdge)
                .where(LineageEdge.output_dataset == output_dataset)
                .order_by(LineageEdge.created_at.desc())
                .limit(limit)
            )
        )

    def document_trace(self, document_id: uuid.UUID) -> dict[str, Any] | None:
        """Live trace: document → versions → current chunk count → embedding coverage hint."""
        doc = self.session.get(Document, document_id)
        if doc is None:
            return None
        versions = list(
            self.session.scalars(
                select(DocumentVersion)
                .where(DocumentVersion.document_id == document_id)
                .order_by(DocumentVersion.fetched_at)
            )
        )
        current_chunks = list(
            self.session.scalars(
                select(Chunk).where(Chunk.document_id == document_id, Chunk.is_current.is_(True))
            )
        )
        return {
            "document_id": str(doc.document_id),
            "canonical_source_uri": doc.canonical_source_uri,
            "status": doc.status,
            "current_version_id": str(doc.current_version_id) if doc.current_version_id else None,
            "is_duplicate_of": str(doc.is_duplicate_of) if doc.is_duplicate_of else None,
            "versions": [
                {
                    "document_version_id": str(v.document_version_id),
                    "content_sha256": v.content_sha256,
                    "bronze_object_key": v.bronze_object_key,
                    "fetched_at": v.fetched_at.isoformat() if v.fetched_at else None,
                    "parsed_at": v.parsed_at.isoformat() if v.parsed_at else None,
                    "parser_version": v.parser_version,
                    "is_current": doc.current_version_id == v.document_version_id,
                }
                for v in versions
            ],
            "current_chunks": len(current_chunks),
            "chunker_versions": sorted({c.chunker_version for c in current_chunks}),
            "embedding_status_counts": {
                status: sum(1 for c in current_chunks if c.embedding_status == status)
                for status in {c.embedding_status for c in current_chunks}
            },
        }
