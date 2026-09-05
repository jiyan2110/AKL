"""DocumentRepository — current-state document/version bookkeeping (PRD §3.7, Appendix A.1/A.2)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from akl import ids
from akl.db.models import Document, DocumentVersion
from akl.db.repositories import Repository


@dataclass(frozen=True)
class BronzeRecordResult:
    document_id: uuid.UUID
    document_version_id: uuid.UUID
    document_created: bool
    version_created: bool


class DocumentRepository(Repository):
    """Upserts and lookups for ``documents`` and ``document_versions``."""

    def record_bronze(
        self,
        *,
        canonical_source_uri: str,
        source_type: str,
        connector_id: str,
        content_sha256: str,
        bronze_object_key: str,
        run_id: str,
        fetched_at: datetime | None = None,
        security_level: str = "internal",
        allowed_groups: Sequence[str] = (),
        title: str | None = None,
        parser_version: str = "",
    ) -> BronzeRecordResult:
        """Register a fetched object. Idempotent per (document, content, parser_version)."""
        document_id = ids.document_id(canonical_source_uri)
        version_id = ids.document_version_id(document_id, content_sha256, parser_version)
        now = fetched_at or datetime.now(UTC)
        # The status expression above is awkward in SQLAlchemy; use a plain follow-up UPDATE for the
        # 'deleted → bronze' resurrection rule instead (clearer and equally atomic within the session).
        doc_stmt = (
            pg_insert(Document)
            .values(
                document_id=document_id,
                canonical_source_uri=canonical_source_uri,
                source_type=source_type,
                connector_id=connector_id,
                title=title,
                status="bronze",
                latest_content_sha256=content_sha256,
                security_level=security_level,
                allowed_groups=list(allowed_groups),
            )
            .on_conflict_do_update(
                index_elements=[Document.document_id],
                set_={
                    "latest_content_sha256": content_sha256,
                    "connector_id": connector_id,
                    "title": func.coalesce(title, Document.title),
                    "deleted_at": None,
                    "updated_at": func.now(),
                },
            )
            .returning(Document.created_at, Document.updated_at)
        )
        created_at, updated_at = self.session.execute(doc_stmt).one()
        document_created = created_at == updated_at
        self.session.execute(
            update(Document)
            .where(
                Document.document_id == document_id, Document.status.in_(["deleted", "deleting"])
            )
            .values(status="bronze")
        )

        ver_stmt = (
            pg_insert(DocumentVersion)
            .values(
                document_version_id=version_id,
                document_id=document_id,
                content_sha256=content_sha256,
                bronze_object_key=bronze_object_key,
                parser_version=parser_version,
                run_id=run_id,
                fetched_at=now,
            )
            .on_conflict_do_nothing(index_elements=[DocumentVersion.document_version_id])
            .returning(DocumentVersion.document_version_id)
        )
        version_created = self.session.execute(ver_stmt).scalar_one_or_none() is not None
        return BronzeRecordResult(document_id, version_id, document_created, version_created)

    def mark_parsed(
        self,
        *,
        document_version_id: uuid.UUID,
        parser_name: str,
        parser_version: str,
        text_sha256: str,
        quality_score: float,
        quality_flags: Sequence[str],
        language: str | None,
        word_count: int,
        silver_partition: str,
        title: str | None,
    ) -> None:
        """Record a successful parse and point the document at this version (status → silver)."""
        version = self.session.get(DocumentVersion, document_version_id)
        if version is None:
            raise LookupError(f"document_version {document_version_id} not found")
        version.parser_name = parser_name
        version.parser_version = parser_version
        version.text_sha256 = text_sha256
        version.quality_score = quality_score
        version.quality_flags = list(quality_flags)
        version.language = language
        version.word_count = word_count
        version.silver_partition = silver_partition
        version.parsed_at = datetime.now(UTC)
        self.session.execute(
            update(Document)
            .where(Document.document_id == version.document_id)
            .values(
                status="silver",
                current_version_id=document_version_id,
                title=func.coalesce(title, Document.title),
                updated_at=func.now(),
            )
        )

    def set_status(self, document_id: uuid.UUID, status: str) -> None:
        values: dict[str, object] = {"status": status, "updated_at": func.now()}
        if status == "deleted":
            values["deleted_at"] = func.now()
        self.session.execute(
            update(Document).where(Document.document_id == document_id).values(**values)
        )

    def get(self, document_id: uuid.UUID) -> Document | None:
        return self.session.get(Document, document_id)

    def get_by_uri(self, canonical_source_uri: str) -> Document | None:
        return self.session.get(Document, ids.document_id(canonical_source_uri))

    def list_by_status(self, status: str, *, limit: int = 1000) -> list[Document]:
        stmt = (
            select(Document)
            .where(Document.status == status)
            .order_by(Document.updated_at, Document.canonical_source_uri)
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

    def versions(self, document_id: uuid.UUID) -> list[DocumentVersion]:
        stmt = (
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.created_at.desc())
        )
        return list(self.session.scalars(stmt))
