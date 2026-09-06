"""Ingestion service (PRD §3.7, §3.8): connector runs, Bronze→Silver parsing with
validation, language detection, quality scoring, near-duplicate detection,
quarantine and deletion handling.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
from sqlalchemy import update
from sqlalchemy.orm import Session

from akl.config import Settings
from akl.db.models import Document
from akl.db.repositories.connector_state import ConnectorStateRepository
from akl.db.repositories.documents import DocumentRepository
from akl.db.repositories.pii import PiiRepository
from akl.db.session import Database
from akl.errors import AKLError
from akl.governance.pii import scan_text
from akl.ingestion.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorRunner,
    IngestionRunReport,
    load_connector_configs,
)
from akl.ingestion.connectors.github import GitHubConnector
from akl.ingestion.connectors.html import HtmlConnector
from akl.ingestion.connectors.markdown import MarkdownConnector
from akl.ingestion.connectors.pdf import PdfConnector
from akl.ingestion.dedup import find_near_duplicates, simhash
from akl.ingestion.language import detect_language, prose_sample
from akl.ingestion.models import FetchedObject, QualityReport, SourceItem, UnifiedDocument
from akl.ingestion.parsers.html import HtmlParser
from akl.ingestion.parsers.markdown import MarkdownParser
from akl.ingestion.parsers.pdf import PdfParser
from akl.ingestion.parsers.text import CodeParser, TextParser
from akl.ingestion.quality import document_quality
from akl.ingestion.quarantine import QuarantineWriter
from akl.ingestion.registry import ConnectorRegistry, ParserRegistry
from akl.ingestion.validators import validate_document, validate_size_upper
from akl.lakehouse.bronze import BronzeStore
from akl.lakehouse.engine import DuckDBEngine
from akl.lakehouse.io import LakehouseIO, Layer
from akl.lakehouse.schemas import enforce
from akl.lakehouse.schemas.bronze import GITHUB_SNAPSHOTS
from akl.lakehouse.silver import SilverStore

DOC_QUALITY_MIN = 0.35  # PRD §3.11 — kept in Silver, excluded from Gold below this


@dataclass
class ParseReport:
    run_id: str
    considered: int = 0
    parsed: int = 0
    skipped: int = 0
    quarantined: int = 0
    duplicates: int = 0
    low_quality: int = 0
    pii_flagged: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)


@dataclass
class DeletionReport:
    documents_tombstoned: int = 0
    chunks_tombstoned: int = 0


def register_builtins(connectors: ConnectorRegistry, parsers: ParserRegistry) -> None:
    for cls in (MarkdownConnector, PdfConnector, HtmlConnector, GitHubConnector):
        if cls.source_type not in connectors:
            connectors.register(cls, cls.config_cls)
    if len(parsers) == 0:
        for parser in (MarkdownParser(), HtmlParser(), PdfParser(), TextParser(), CodeParser()):
            parsers.register(parser)


class IngestionService:
    def __init__(self, settings: Settings, engine: DuckDBEngine, db: Database) -> None:
        self.settings = settings
        self.engine = engine
        self.db = db
        self.io = LakehouseIO(settings, engine)
        self.bronze = BronzeStore(self.io)
        self.silver = SilverStore(self.io, engine)
        self.connectors = ConnectorRegistry()
        self.parsers = ParserRegistry()
        register_builtins(self.connectors, self.parsers)

    # -- configs ------------------------------------------------------------------------
    def connector_configs(self) -> list[ConnectorConfig]:
        directory = Path(self.settings.core.config_dir) / "connectors"
        if not directory.is_dir():
            return []
        return [
            c for c in load_connector_configs(directory, self.connectors.config_types) if c.enabled
        ]

    def connector_config(self, connector_id: str) -> ConnectorConfig:
        for cfg in self.connector_configs():
            if cfg.id == connector_id:
                return cfg
        raise KeyError(f"connector {connector_id!r} not found or disabled in configs/connectors")

    # -- fetch → bronze --------------------------------------------------------------------
    def run_connector(
        self,
        connector_id: str | None = None,
        *,
        run_id: str,
        connector: BaseConnector | None = None,
    ) -> IngestionRunReport:
        """Discover/fetch to Bronze, record documents, persist state, apply deletions."""
        if connector is None:
            if connector_id is None:
                raise ValueError("connector_id or connector required")
            connector = self.connectors.create(self.connector_config(connector_id))
        cfg = connector.config
        with self.db.session() as session:
            state_repo = ConnectorStateRepository(session)
            runner = ConnectorRunner(self.bronze, DocumentRepository(session))
            report = runner.run(connector, state=state_repo.get(cfg.id), run_id=run_id)
            state_repo.save(cfg.id, connector.name, report.new_state, run_id=run_id)
            if report.failed == 0:
                state_repo.mark_success(
                    cfg.id, documents_count=self._count_documents(report.new_state)
                )
            if report.deletions:
                self.apply_deletions(
                    [d.canonical_uri for d in report.deletions], run_id=run_id, session=session
                )
        snapshot_rows = getattr(connector, "snapshot_rows", None)
        if snapshot_rows:
            self._write_github_snapshot(snapshot_rows, run_id=run_id)
        return report

    @staticmethod
    def _count_documents(state: dict[str, Any]) -> int:
        for key in ("files", "urls", "tree"):
            if isinstance(state.get(key), dict):
                return len(state[key])
        return 0

    def _write_github_snapshot(self, rows: list[dict[str, Any]], *, run_id: str) -> None:
        normalised = [
            {**r, "snapshot_at": datetime.fromtimestamp(float(r["snapshot_at"]), tz=UTC)}
            for r in rows
        ]
        table = enforce(
            pa.Table.from_pylist(normalised, schema=GITHUB_SNAPSHOTS.schema), GITHUB_SNAPSHOTS
        )
        self.io.write(
            table,
            Layer.BRONZE,
            "github_snapshots",
            run_id=run_id,
            schema_version=GITHUB_SNAPSHOTS.version,
            partition_by=GITHUB_SNAPSHOTS.partition_by,
            sort_by=GITHUB_SNAPSHOTS.sort_by,
        )

    # -- deletions -------------------------------------------------------------------------
    def apply_deletions(
        self, canonical_uris: list[str], *, run_id: str, session: Session | None = None
    ) -> DeletionReport:
        """Soft-delete documents that vanished at the source (PRD §9.11 tombstone flow)."""
        report = DeletionReport()
        if not canonical_uris:
            return report

        def _apply(s: Session) -> None:
            repo = DocumentRepository(s)
            ids_to_delete: list[str] = []
            for uri in canonical_uris:
                doc = repo.get_by_uri(uri)
                if doc is not None and doc.status not in ("deleted", "deleting"):
                    ids_to_delete.append(str(doc.document_id))
            if not ids_to_delete:
                return
            docs, chunks = self.silver.tombstone_documents(ids_to_delete, run_id=run_id)
            report.documents_tombstoned, report.chunks_tombstoned = docs, chunks
            for doc_id in ids_to_delete:
                repo.set_status(uuid.UUID(doc_id), "deleted")

        if session is not None:
            _apply(session)
        else:
            with self.db.session() as s:
                _apply(s)
        return report

    # -- bronze → silver ---------------------------------------------------------------------
    def parse_backlog(
        self, *, run_id: str, limit: int = 500, allow_secret_like: bool = False
    ) -> ParseReport:
        report = ParseReport(run_id=run_id)
        rows: list[dict[str, Any]] = []
        marks: list[dict[str, Any]] = []
        candidates: list[tuple[str, int, float]] = []
        now = datetime.now(UTC)

        with self.db.session() as session:
            repo = DocumentRepository(session)
            quarantine = QuarantineWriter(self.io, session)
            for doc in repo.list_by_status("bronze", limit=limit):
                report.considered += 1
                versions = repo.versions(doc.document_id)
                if not versions:
                    continue
                latest = versions[0]
                unified = self._parse_one(
                    doc, latest, repo, quarantine, report, run_id, allow_secret_like
                )
                if unified is None:
                    continue
                parser_version = unified.parser_version
                if any(
                    v.content_sha256 == latest.content_sha256
                    and v.parser_version == parser_version
                    and v.parsed_at
                    for v in versions
                ):
                    report.skipped += 1
                    continue
                row = SilverStore.prepare_document_row(unified.to_silver_row(parsed_at=now))
                if unified.quality.score < DOC_QUALITY_MIN:
                    row["quality_flags"] = [*row["quality_flags"], "low_quality"]
                    report.low_quality += 1
                rows.append(row)
                candidates.append(
                    (
                        row["document_id"],
                        int(row["fingerprint_simhash"] or 0),
                        float(row["quality_score"]),
                    )
                )
                recorded = repo.record_bronze(
                    canonical_source_uri=doc.canonical_source_uri,
                    source_type=doc.source_type,
                    connector_id=doc.connector_id,
                    content_sha256=latest.content_sha256,
                    bronze_object_key=latest.bronze_object_key,
                    run_id=run_id,
                    fetched_at=latest.fetched_at,
                    security_level=unified.security_level,
                    allowed_groups=unified.allowed_groups,
                    parser_version=parser_version,
                )
                if self.settings.governance.pii_scan_enabled:
                    findings = scan_text(
                        unified.text,
                        enabled_types=frozenset(self.settings.governance.pii_types_enabled),
                    )
                    if findings:
                        PiiRepository(session).record(
                            document_id=recorded.document_id, findings=findings
                        )
                        report.pii_flagged += 1
                marks.append(
                    {
                        "document_version_id": recorded.document_version_id,
                        "document_id": doc.document_id,
                        "parser_name": unified.parser_name,
                        "parser_version": parser_version,
                        "text_sha256": unified.text_sha256,
                        "quality_score": row["quality_score"],
                        "quality_flags": list(row["quality_flags"]),
                        "language": unified.language,
                        "word_count": row["word_count"],
                        "silver_partition": f"source_type={unified.source_type}/ingest_date={now.date().isoformat()}",
                        "title": unified.title,
                    }
                )

            # near-duplicate detection against current Silver documents (PRD §2.5.3)
            ledger_rows: list[dict[str, Any]] = []
            if rows:
                existing = self._existing_fingerprints(exclude={r["document_id"] for r in rows})
                uri_by_id = {r["document_id"]: r["canonical_source_uri"] for r in rows}
                candidates.sort(
                    key=lambda c: uri_by_id.get(c[0], "")
                )  # deterministic in-batch "earliest"
                decisions = find_near_duplicates(candidates, existing)
                by_dup = {d.duplicate_document_id: d for d in decisions}
                for row in rows:
                    decision = by_dup.get(row["document_id"])
                    if decision is not None:
                        row["is_duplicate_of"] = decision.canonical_document_id
                        report.duplicates += 1
                        ledger_rows.append(
                            {
                                "fingerprint_simhash": decision.fingerprint,
                                "canonical_document_id": decision.canonical_document_id,
                                "duplicate_document_id": decision.duplicate_document_id,
                                "hamming_distance": decision.hamming_distance,
                                "decided_at": now,
                            }
                        )
                self.silver.write_documents(rows, run_id=run_id)  # Parquet first, then pointers
                if ledger_rows:
                    self.silver.write_dedup_ledger(ledger_rows, run_id=run_id)
                for m in marks:
                    doc_id = m.pop("document_id")
                    repo.mark_parsed(**m)
                    decision = by_dup.get(str(doc_id))
                    if decision is not None:
                        session.execute(
                            update(Document)
                            .where(Document.document_id == doc_id)
                            .values(is_duplicate_of=uuid.UUID(decision.canonical_document_id))
                        )
                report.parsed = len(rows)
            quarantine.flush(run_id=run_id)
        return report

    def _parse_one(
        self,
        doc: Document,
        latest: Any,
        repo: DocumentRepository,
        quarantine: QuarantineWriter,
        report: ParseReport,
        run_id: str,
        allow_secret_like: bool,
    ) -> UnifiedDocument | None:
        def fail(code: str, stage: str, detail: str) -> None:
            quarantine.quarantine(
                document_id=doc.document_id,
                content_sha256=latest.content_sha256,
                source_type=doc.source_type,
                bronze_object_key=latest.bronze_object_key,
                error_code=code,
                stage=stage,
                detail=detail,
                run_id=run_id,
            )
            report.quarantined += 1
            report.failures.append(
                {"document_id": str(doc.document_id), "code": code, "error": detail[:200]}
            )

        ext = latest.bronze_object_key.rsplit(".", 1)[-1]
        try:
            data = self.bronze.get_raw(latest.bronze_object_key)
        except AKLError as exc:
            fail(exc.code, "fetch_bronze", exc.message)
            return None
        size_check = validate_size_upper(len(data))  # oversized objects are never parsed
        if size_check.rejected:
            fail("AKL-E3001", "validate", size_check.detail)
            return None
        try:
            parser = self.parsers.select(None, ext, doc.source_type)
        except AKLError as exc:
            fail(exc.code, "select_parser", exc.message)
            return None
        item = SourceItem(
            uri=doc.canonical_source_uri,
            canonical_uri=doc.canonical_source_uri,
            source_type=doc.source_type,
            filename=f"document.{ext}",
            security_level=doc.security_level,
            allowed_groups=tuple(doc.allowed_groups),
            source_metadata={k: str(v) for k, v in (doc.metadata_ or {}).items()},
        )
        try:
            unified = parser.parse(
                FetchedObject.from_bytes(item, data, mime_type=None, fetched_at=latest.fetched_at)
            )
        except AKLError as exc:
            fail(exc.code, "parse", exc.message)
            return None
        except Exception as exc:  # defensive: parser bug must not kill the run
            fail("AKL-E3030", "parse", f"{type(exc).__name__}: {exc}")
            return None

        result = validate_document(
            unified, size_bytes=len(data), allow_secret_like=allow_secret_like
        )
        if result.rejected:
            fail(result.reject_code or "AKL-E3005", "validate", result.detail)
            return None
        lang, conf = detect_language(prose_sample(unified.text, unified.blocks))
        flags = tuple(
            dict.fromkeys(
                [
                    *unified.quality.flags,
                    *result.flags,
                    *(("low_language_confidence",) if lang == "und" else ()),
                ]
            )
        )
        score = document_quality(unified, language_confidence=conf)
        return unified.model_copy(
            update={
                "language": lang,
                "quality": QualityReport(score=score, flags=flags),
                "fingerprint_simhash": simhash(unified.text),
            }
        )

    def _existing_fingerprints(self, *, exclude: set[str]) -> list[tuple[str, int, float]]:
        try:
            table = self.silver.current_documents(
                columns=["document_id", "fingerprint_simhash", "quality_score"],
                where="fingerprint_simhash IS NOT NULL AND is_duplicate_of IS NULL",
            )
        except AKLError:
            return []
        return [
            (str(r["document_id"]), int(r["fingerprint_simhash"]), float(r["quality_score"]))
            for r in table.to_pylist()
            if str(r["document_id"]) not in exclude
        ]
