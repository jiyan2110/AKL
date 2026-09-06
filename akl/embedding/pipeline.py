"""Embedding pipeline (PRD §5.1–5.7): backlog → cache → generate → Gold → Postgres status."""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from akl.config import Settings
from akl.db.repositories.chunks import ChunkRepository
from akl.db.repositories.embedding_cache import EmbeddingCacheRepository
from akl.db.session import Database
from akl.embedding.provider import EmbeddingProvider, build_provider
from akl.lakehouse.engine import DuckDBEngine
from akl.lakehouse.gold import GoldStore
from akl.lakehouse.io import LakehouseIO
from akl.observability.mlflow_tracking import log_metrics, log_params, mlflow_run


@dataclass
class EmbeddingRunReport:
    run_id: str
    embedding_version: str
    backlog: int = 0
    cache_hits: int = 0
    generated: int = 0
    written: int = 0
    failed: int = 0
    batches: int = 0
    duration_s: float = 0.0
    throughput_cps: float = 0.0
    job_id: uuid.UUID | None = None
    failures: list[dict[str, str]] = field(default_factory=list)


class EmbeddingPipeline:
    """Embeds every active Gold unit lacking a vector for the configured embedding version."""

    def __init__(
        self,
        settings: Settings,
        engine: DuckDBEngine,
        db: Database,
        *,
        provider: EmbeddingProvider | None = None,
        allow_download: bool = True,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.db = db
        self.io = LakehouseIO(settings, engine)
        self.provider = provider or build_provider(
            settings.embedding, settings.core.models_dir, allow_download=allow_download
        )
        self.embedding_version = self.provider.embedding_version
        self.gold = GoldStore(
            self.io,
            engine,
            embedding_version=self.embedding_version,
            embedding_dim=self.provider.dim,
            view_params={"chunker_version": settings.chunking.chunker_version},
        )

    def backlog(self, *, document_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Active units with no/stale vector, joined to the text that must be embedded."""
        self.gold.ensure_views(refresh=True)
        scope = ""
        if document_ids:
            ids_sql = ", ".join(f"'{d}'" for d in document_ids)
            scope = f" AND u.document_id IN ({ids_sql})"
        rows: list[dict[str, Any]] = self.engine.execute(
            f"""
            SELECT u.chunk_id, u.chunk_checksum, u.embedded_text_sha256, u.source_type,
                   coalesce(u.context_prefix, '') AS context_prefix, u.text
            FROM v_embedding_coverage AS cov
            JOIN v_gold_active_units AS u ON u.chunk_id = cov.chunk_id
            WHERE (NOT cov.has_embedding OR cov.stale_embedding){scope}
            ORDER BY u.document_id, u.chunk_index
            """  # noqa: S608 - uuids only
        ).to_pylist()
        return rows

    def run(
        self,
        *,
        run_id: str,
        limit: int | None = None,
        batch_size: int | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> EmbeddingRunReport:
        with mlflow_run(
            self.settings.observability,
            run_name=run_id,
            tags={"embedding_version": self.embedding_version},
        ) as mlrun:
            log_params(
                mlrun,
                {
                    "provider": self.settings.embedding.embed_provider,
                    "model_id": self.provider.model_id,
                    "embedding_version": self.embedding_version,
                    "batch_size": batch_size or self.settings.embedding.embed_batch_size,
                    "limit": limit,
                },
            )
            report = self._run(
                run_id=run_id, limit=limit, batch_size=batch_size, document_ids=document_ids
            )
            log_metrics(
                mlrun,
                {
                    "backlog": report.backlog,
                    "cache_hits": report.cache_hits,
                    "generated": report.generated,
                    "written": report.written,
                    "failed": report.failed,
                    "duration_s": report.duration_s,
                    "throughput_cps": report.throughput_cps,
                },
            )
            return report

    def _run(
        self,
        *,
        run_id: str,
        limit: int | None = None,
        batch_size: int | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> EmbeddingRunReport:
        start = time.perf_counter()
        report = EmbeddingRunReport(run_id=run_id, embedding_version=self.embedding_version)
        items = self.backlog(document_ids=document_ids)
        if limit is not None:
            items = items[:limit]
        report.backlog = len(items)
        if not items:
            report.duration_s = time.perf_counter() - start
            return report

        # embedded text = context_prefix + "\n" + text (PRD §4.8); sort by length to reduce padding waste
        for it in items:
            it["embedded_text"] = (
                f"{it['context_prefix']}\n{it['text']}" if it["context_prefix"] else it["text"]
            )
        items.sort(key=lambda r: len(r["embedded_text"]))
        size = batch_size or self.settings.embedding.embed_batch_size
        now = datetime.now(UTC)
        model_id, model_version, dim = (
            self.provider.model_id,
            self.provider.model_version,
            self.provider.dim,
        )

        gold_rows: list[dict[str, Any]] = []
        embedded_ids: list[uuid.UUID] = []
        failed_ids: list[uuid.UUID] = []
        with self.db.session() as session:
            cache = EmbeddingCacheRepository(session)
            for start_idx in range(0, len(items), size):
                batch = items[start_idx : start_idx + size]
                report.batches += 1
                shas = [b["embedded_text_sha256"] for b in batch]
                hits = cache.lookup(shas, model_id=model_id, model_version=model_version, dim=dim)
                misses = [b for b in batch if b["embedded_text_sha256"] not in hits]
                vectors: dict[str, np.ndarray] = dict(hits)
                report.cache_hits += len(hits)
                if misses:
                    try:
                        matrix = self.provider.embed_documents([m["embedded_text"] for m in misses])
                    except Exception as exc:  # AKLError or unexpected — batch fails, run continues
                        report.failed += len(misses)
                        failed_ids.extend(uuid.UUID(str(m["chunk_id"])) for m in misses)
                        report.failures.append(
                            {"batch": str(report.batches), "error": str(exc)[:200]}
                        )
                        matrix = None
                    if matrix is not None:
                        for m, vec in zip(misses, matrix, strict=True):
                            if not np.all(np.isfinite(vec)):
                                report.failed += 1
                                failed_ids.append(uuid.UUID(str(m["chunk_id"])))
                                continue
                            vectors[m["embedded_text_sha256"]] = vec
                        cache.store(
                            (
                                (m["embedded_text_sha256"], vectors[m["embedded_text_sha256"]])
                                for m in misses
                                if m["embedded_text_sha256"] in vectors
                            ),
                            model_id=model_id,
                            model_version=model_version,
                            dim=dim,
                        )
                        report.generated += sum(
                            1 for m in misses if m["embedded_text_sha256"] in vectors
                        )
                for b in batch:
                    vec = vectors.get(b["embedded_text_sha256"])
                    if vec is None:
                        continue
                    gold_rows.append(
                        {
                            "chunk_id": str(b["chunk_id"]),
                            "chunk_checksum": b["chunk_checksum"],
                            "embedded_text_sha256": b["embedded_text_sha256"],
                            "embedding_version": self.embedding_version,
                            "model_id": model_id,
                            "model_version": model_version,
                            "dim": dim,
                            "vector": [float(x) for x in vec],
                            "embedded_at": now,
                            "embedder_version": self.settings.embedding.embedder_version,
                            "mlflow_run_id": None,
                            "source_type": b["source_type"],
                        }
                    )
                    embedded_ids.append(uuid.UUID(str(b["chunk_id"])))
            if gold_rows:
                self.gold.write_embeddings(
                    gold_rows, run_id=run_id
                )  # Parquet first (durable record)
                report.written = len(gold_rows)
            chunks = ChunkRepository(session)
            chunks.set_embedding_status(embedded_ids, "embedded")
            chunks.set_embedding_status(failed_ids, "failed")
            report.duration_s = time.perf_counter() - start
            report.throughput_cps = (
                round(report.written / report.duration_s, 1) if report.duration_s else 0.0
            )
            report.job_id = cache.record_job(
                run_id=run_id,
                embedding_version=self.embedding_version,
                shard=0,
                chunks_total=report.backlog,
                cache_hits=report.cache_hits,
                generated=report.generated,
                failed=report.failed,
                started_at=now,
                finished_at=datetime.now(UTC),
                throughput_cps=report.throughput_cps,
            )
        return report

    def coverage(self) -> tuple[float, int]:
        return self.gold.coverage_ratio(), self.gold.embedding_backlog().num_rows
