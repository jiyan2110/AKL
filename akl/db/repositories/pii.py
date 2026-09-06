"""PiiRepository — writes/reads ``pii_mentions`` (PRD §9.6). Only ``value_hash`` is ever stored."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete, func, insert, select

from akl.db.models import PiiMention
from akl.db.repositories import Repository
from akl.governance.pii import PiiFinding


def hash_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class PiiRepository(Repository):
    def record(
        self,
        *,
        document_id: uuid.UUID,
        findings: Sequence[PiiFinding],
        chunk_id: uuid.UUID | None = None,
    ) -> int:
        if not findings:
            return 0
        rows = [
            {
                "document_id": document_id,
                "chunk_id": chunk_id,
                "pii_type": f.pii_type,
                "value_hash": hash_value(f.value),
            }
            for f in findings
        ]
        self.session.execute(insert(PiiMention), rows)
        return len(rows)

    def for_document(self, document_id: uuid.UUID) -> list[PiiMention]:
        return list(
            self.session.scalars(
                select(PiiMention)
                .where(PiiMention.document_id == document_id)
                .order_by(PiiMention.id)
            )
        )

    def counts_by_type(self, document_id: uuid.UUID) -> dict[str, int]:
        rows = self.session.execute(
            select(PiiMention.pii_type, func.count())
            .where(PiiMention.document_id == document_id)
            .group_by(PiiMention.pii_type)
        ).all()
        return {str(t): int(c) for t, c in rows}

    def delete_for_document(self, document_id: uuid.UUID) -> int:
        result = self.session.execute(
            delete(PiiMention).where(PiiMention.document_id == document_id)
        )
        return int(getattr(result, "rowcount", 0) or 0)

    def any_document_has_pii(self, document_ids: Sequence[uuid.UUID]) -> dict[str, Any]:
        if not document_ids:
            return {}
        rows = self.session.execute(
            select(PiiMention.document_id, func.count())
            .where(PiiMention.document_id.in_(document_ids))
            .group_by(PiiMention.document_id)
        ).all()
        return {str(d): int(c) for d, c in rows}
