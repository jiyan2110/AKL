"""Local post-ingest pipeline: parse → chunk → embed → Qdrant sync → BM25 (PRD §10.3 async upload).

Runs the same service calls the Airflow DAGs will orchestrate, in-process, for API-triggered work
until orchestration lands (Batch F). Safe to call concurrently for different documents; each
stage is incremental and idempotent.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from akl.chunking.incremental import ChunkingService
from akl.config import Settings
from akl.db.session import Database
from akl.embedding.bm25.builder import build_bm25
from akl.embedding.pipeline import EmbeddingPipeline
from akl.embedding.qdrant.reconciler import QdrantReconciler
from akl.embedding.qdrant.schema import make_client
from akl.errors import AKLError
from akl.ingestion.service import IngestionService
from akl.lakehouse.bronze import new_run_id
from akl.lakehouse.engine import DuckDBEngine

_LOCK = (
    threading.Lock()
)  # one local pipeline at a time per process (DuckDB engine is not thread-safe)


@dataclass
class LocalPipelineReport:
    run_id: str
    parsed: int = 0
    quarantined: int = 0
    chunked: int = 0
    embedded: int = 0
    qdrant_upserted: int = 0
    qdrant_deleted: int = 0
    bm25_documents: int | None = None
    stages: list[str] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def run_post_ingest(
    settings: Settings,
    db: Database,
    *,
    document_ids: Sequence[uuid.UUID] | None = None,
    run_id: str | None = None,
    embed: bool = True,
    sync: bool = True,
    rebuild_bm25: bool = True,
    on_stage: Any = None,
) -> LocalPipelineReport:
    """Parse the Bronze backlog, chunk, embed and reconcile Qdrant; returns a stage-by-stage report."""
    run_id = run_id or new_run_id("api")
    report = LocalPipelineReport(run_id=run_id)
    with _LOCK, DuckDBEngine(settings) as engine:
        try:
            ingest = IngestionService(settings, engine, db)
            parsed = ingest.parse_backlog(run_id=run_id)
            report.parsed, report.quarantined = parsed.parsed, parsed.quarantined
            report.stages.append("parse")
            _notify(on_stage, "parse", report)

            chunking = ChunkingService(settings, engine, db)
            chunk_rep = chunking.run(run_id=run_id, document_ids=document_ids)
            report.chunked = chunk_rep.documents_chunked
            report.stages.append("chunk")
            _notify(on_stage, "chunk", report)

            if embed:
                pipeline = EmbeddingPipeline(settings, engine, db)
                emb = pipeline.run(
                    run_id=run_id,
                    document_ids=[str(d) for d in document_ids] if document_ids else None,
                )
                report.embedded = emb.written
                report.stages.append("embed")
                _notify(on_stage, "embed", report)
                if sync:
                    reconciler = QdrantReconciler(
                        make_client(settings), settings, engine, pipeline.gold
                    )
                    s = reconciler.sync(run_id=run_id)
                    report.qdrant_upserted, report.qdrant_deleted = s.upserted, s.deleted
                    report.stages.append("qdrant_sync")
                    _notify(on_stage, "qdrant_sync", report)
                    if rebuild_bm25:
                        b = build_bm25(settings, pipeline.io, pipeline.gold, version=run_id)
                        report.bm25_documents = b.documents
                        report.stages.append("bm25")
                        _notify(on_stage, "bm25", report)
        except AKLError as exc:
            report.errors.append(
                {
                    "code": exc.code,
                    "error": exc.message,
                    "stage": report.stages[-1] if report.stages else "start",
                }
            )
        except Exception as exc:  # never let a background job die silently
            report.errors.append(
                {
                    "code": "AKL-E7001",
                    "error": f"{type(exc).__name__}: {exc}",
                    "stage": report.stages[-1] if report.stages else "start",
                }
            )
    return report


def _notify(cb: Any, stage: str, report: LocalPipelineReport) -> None:
    if cb is not None:
        cb(stage, report)
