"""ChunkRepository — current-state chunk index and chunking backlog (PRD Appendix A.3, §4.12)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import and_, exists, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from akl.db.models import Chunk, Document, DocumentVersion
from akl.db.repositories import Repository


class ChunkRepository(Repository):
    def current_for_document(self, document_id: uuid.UUID) -> list[Chunk]:
        stmt = (
            select(Chunk)
            .where(
                Chunk.document_id == document_id,
                Chunk.is_current.is_(True),
                Chunk.is_deleted.is_(False),
            )
            .order_by(Chunk.chunk_index)
        )
        return list(self.session.scalars(stmt))

    def upsert_current(self, rows: Sequence[dict[str, Any]]) -> int:
        """Insert or re-activate chunks; returns the number of rows sent."""
        if not rows:
            return 0
        stmt = pg_insert(Chunk).values(list(rows))
        stmt = stmt.on_conflict_do_update(
            index_elements=[Chunk.chunk_id],
            set_={
                "lineage_id": stmt.excluded.lineage_id,
                "document_version_id": stmt.excluded.document_version_id,
                "chunk_index": stmt.excluded.chunk_index,
                "embedded_text_sha256": stmt.excluded.embedded_text_sha256,
                "token_count": stmt.excluded.token_count,
                "quality_score": stmt.excluded.quality_score,
                "security_level": stmt.excluded.security_level,
                "allowed_groups": stmt.excluded.allowed_groups,
                "chunker_version": stmt.excluded.chunker_version,
                "chunk_config_hash": stmt.excluded.chunk_config_hash,
                "is_current": True,
                "is_deleted": False,
                "updated_at": func.now(),
            },
        )
        self.session.execute(stmt)
        return len(rows)

    def retire(self, chunk_ids: Sequence[uuid.UUID], *, deleted: bool) -> int:
        if not chunk_ids:
            return 0
        result = self.session.execute(
            update(Chunk)
            .where(Chunk.chunk_id.in_(list(chunk_ids)))
            .values(is_current=False, is_deleted=deleted, updated_at=func.now())
        )
        return int(getattr(result, "rowcount", 0) or 0)

    def set_embedding_status(self, chunk_ids: Sequence[uuid.UUID], status: str) -> int:
        if not chunk_ids:
            return 0
        result = self.session.execute(
            update(Chunk)
            .where(Chunk.chunk_id.in_(list(chunk_ids)))
            .values(embedding_status=status, updated_at=func.now())
        )
        return int(getattr(result, "rowcount", 0) or 0)

    def documents_needing_chunks(
        self,
        *,
        chunker_version: str,
        chunk_config_hash: str,
        limit: int = 200,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[tuple[Document, DocumentVersion]]:
        """Current document versions with no current chunks for this chunker version/config."""
        has_chunks = exists().where(
            and_(
                Chunk.document_version_id == Document.current_version_id,
                Chunk.chunker_version == chunker_version,
                Chunk.chunk_config_hash == chunk_config_hash,
                Chunk.is_current.is_(True),
            )
        )
        stmt = (
            select(Document, DocumentVersion)
            .join(
                DocumentVersion, DocumentVersion.document_version_id == Document.current_version_id
            )
            .where(
                Document.status.in_(["silver", "gold"]),
                Document.is_duplicate_of.is_(None),
                ~has_chunks,
            )
            .order_by(Document.updated_at, Document.canonical_source_uri)
            .limit(limit)
        )
        if document_ids:
            stmt = stmt.where(Document.document_id.in_(list(document_ids)))
        return [(d, v) for d, v in self.session.execute(stmt).all()]

    def counts(self) -> dict[str, int]:
        total = int(self.session.scalar(select(func.count()).select_from(Chunk)) or 0)
        current = int(
            self.session.scalar(
                select(func.count()).select_from(Chunk).where(Chunk.is_current.is_(True))
            )
            or 0
        )
        pending = int(
            self.session.scalar(
                select(func.count())
                .select_from(Chunk)
                .where(Chunk.is_current.is_(True), Chunk.embedding_status == "pending")
            )
            or 0
        )
        return {"total": total, "current": current, "embedding_pending": pending}
