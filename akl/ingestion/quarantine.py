"""Quarantine flow (PRD §3.6): failed inputs are copied to ``quarantine/``, a reason row
is appended, a ``quarantine_items`` row is inserted and the document is marked ``quarantined``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa
from sqlalchemy.orm import Session

from akl.db.models import QuarantineItem
from akl.db.repositories.documents import DocumentRepository
from akl.lakehouse.io import LakehouseIO, Layer
from akl.lakehouse.schemas import DatasetSchema, enforce

QUARANTINE_REASONS = DatasetSchema(
    name="quarantine/reasons",
    version="1.0.0",
    description="One row per quarantined object with the failing rule/stage.",
    partition_by=("ingest_date",),
    sort_by=("created_at",),
    schema=pa.schema(
        [
            pa.field("quarantine_id", pa.string(), nullable=False),
            pa.field("document_id", pa.string(), nullable=True),
            pa.field("content_sha256", pa.string(), nullable=True),
            pa.field("source_type", pa.string(), nullable=True),
            pa.field("error_code", pa.string(), nullable=False),
            pa.field("stage", pa.string(), nullable=False),
            pa.field("detail", pa.string(), nullable=True),
            pa.field("bronze_object_key", pa.string(), nullable=True),
            pa.field("quarantine_object_key", pa.string(), nullable=True),
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("ingest_date", pa.date32(), nullable=False),
        ]
    ),
)


@dataclass(frozen=True)
class QuarantineReceipt:
    quarantine_id: uuid.UUID
    error_code: str
    stage: str
    quarantine_object_key: str | None


class QuarantineWriter:
    """Buffers reason rows for one run; call :meth:`flush` at the end of the run."""

    def __init__(self, io: LakehouseIO, session: Session) -> None:
        self._io = io
        self._session = session
        self._rows: list[dict[str, Any]] = []

    def quarantine(
        self,
        *,
        document_id: uuid.UUID | None,
        content_sha256: str | None,
        source_type: str | None,
        bronze_object_key: str | None,
        error_code: str,
        stage: str,
        detail: str,
        run_id: str,
        copy_object: bool = True,
    ) -> QuarantineReceipt:
        now = datetime.now(UTC)
        qid = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"akl:quarantine:{document_id}:{content_sha256}:{error_code}:{run_id}",
        )
        q_key: str | None = None
        if copy_object and bronze_object_key and content_sha256:
            ext = bronze_object_key.rsplit(".", 1)[-1]
            q_key = f"quarantine/ingest_date={now.date().isoformat()}/objects/sha256={content_sha256}.{ext}"
            if not self._io.object_exists(q_key):
                self._io.put_object(q_key, self._io.get_object(bronze_object_key))
        self._rows.append(
            {
                "quarantine_id": str(qid),
                "document_id": str(document_id) if document_id else None,
                "content_sha256": content_sha256,
                "source_type": source_type,
                "error_code": error_code,
                "stage": stage,
                "detail": detail[:2000],
                "bronze_object_key": bronze_object_key,
                "quarantine_object_key": q_key,
                "run_id": run_id,
                "created_at": now,
                "ingest_date": now.date(),
            }
        )
        self._session.merge(
            QuarantineItem(
                quarantine_id=qid,
                document_id=document_id,
                content_sha256=content_sha256,
                source_type=source_type,
                error_code=error_code,
                stage=stage,
                detail=detail[:2000],
                object_key=q_key or bronze_object_key,
                run_id=run_id,
                status="open",
            )
        )
        if document_id is not None:
            DocumentRepository(self._session).set_status(document_id, "quarantined")
        return QuarantineReceipt(qid, error_code, stage, q_key)

    def flush(self, *, run_id: str) -> int:
        if not self._rows:
            return 0
        table = enforce(
            pa.Table.from_pylist(self._rows, schema=QUARANTINE_REASONS.schema), QUARANTINE_REASONS
        )
        self._io.write(
            table,
            Layer.QUARANTINE,
            "reasons",
            run_id=run_id,
            schema_version=QUARANTINE_REASONS.version,
            partition_by=QUARANTINE_REASONS.partition_by,
            sort_by=QUARANTINE_REASONS.sort_by,
        )
        n = len(self._rows)
        self._rows.clear()
        return n
