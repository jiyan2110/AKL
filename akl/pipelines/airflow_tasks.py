"""Pipeline task entrypoints for Airflow (PRD Chapter 7).

Every function here is a thin, JSON-in/JSON-out unit of work: build settings from the
environment, open the engine/database, call the service layer, record ``pipeline_runs`` /
``task_runs``, and return a plain dict (XCom-safe). DAG files contain no business logic.

Quality gates (PRD §7.8) raise :class:`GateFailed` so the Airflow task fails and downstream
Dataset publishing does not happen.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from akl.config import Settings
from akl.db.repositories.lineage import LineageRepository
from akl.db.repositories.runs import RunRepository
from akl.db.session import Database
from akl.errors import AKLError
from akl.lakehouse.engine import DuckDBEngine
from akl.observability.logging import bind_context, get_logger
from akl.observability.metrics import PipelineMetrics, apply_task_metrics

log = get_logger("akl.pipelines")


class GateFailed(AKLError):  # noqa: N818 - matches PRD error naming (AKL-E7001 quality gate)
    code = "AKL-E7001"
    retryable = False


@dataclass
class TaskContext:
    settings: Settings
    engine: DuckDBEngine
    db: Database
    run_id: str
    dag_id: str
    task_id: str
    lineage: dict[str, Any] | None = None


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, int | float) else None


@contextmanager
def task_scope(
    run_id: str,
    dag_id: str,
    task_id: str,
    *,
    map_index: int = -1,
    try_number: int = 1,
    conf: dict[str, Any] | None = None,
) -> Iterator[TaskContext]:
    """Open resources, record the task in Postgres, close everything afterwards."""
    settings = Settings.load()
    db = Database(settings)
    with db.session() as s:
        repo = RunRepository(s)
        repo.start_run(run_id, dag_id, conf=conf)
        task_pk = repo.start_task(run_id, task_id, map_index=map_index, try_number=try_number)
    start = time.perf_counter()
    engine = DuckDBEngine(settings)
    ctx = TaskContext(settings, engine, db, run_id, dag_id, task_id)
    pm = PipelineMetrics()
    state = "success"
    metrics: dict[str, Any] = {}
    with bind_context(
        run_id=run_id,
        dag_id=dag_id,
        task_id=task_id,
        map_index=map_index if map_index != -1 else None,
    ):
        gate_failed = False
        try:
            log.info("task_started")
            yield ctx
            metrics = getattr(ctx, "metrics", {}) or {}
        except GateFailed as exc:
            state, gate_failed = "failed", True
            metrics = {"error": f"{type(exc).__name__}: {exc}"[:500]}
            log.error("task_failed", error=metrics["error"], gate=True)
            raise
        except Exception as exc:
            state = "failed"
            metrics = {"error": f"{type(exc).__name__}: {exc}"[:500]}
            log.error("task_failed", error=metrics["error"], exc_info=True)
            raise
        finally:
            engine.close()
            duration_s = round(time.perf_counter() - start, 2)
            apply_task_metrics(pm, dag_id=dag_id, task_id=task_id, out=metrics)
            pm.task_duration.labels(dag_id=dag_id, task_id=task_id, state=state).observe(duration_s)
            if gate_failed:
                pm.gate_failures.labels(dag_id=dag_id, gate=task_id).inc()
            try:
                with db.session() as s:
                    RunRepository(s).finish_task(
                        task_pk,
                        state=state,
                        rows_in=_int_or_none(metrics.get("rows_in")),
                        rows_out=_int_or_none(metrics.get("rows_out")),
                        metrics={**metrics, "duration_s": duration_s},
                    )
                    if ctx.lineage and settings.observability.lineage_enabled:
                        LineageRepository(s).record(run_id=run_id, task_id=task_id, **ctx.lineage)
            finally:
                db.dispose()
            pm.push(
                settings.observability.pushgateway_url,
                job=settings.observability.pushgateway_job,
                dag_id=dag_id,
                task_id=task_id,
            )
            log.info("task_finished", state=state, duration_s=duration_s)


def finish_run(
    run_id: str, dag_id: str, *, state: str = "success", gold_snapshot_id: str | None = None
) -> dict[str, Any]:
    settings = Settings.load()
    db = Database(settings)
    try:
        with db.session() as s:
            RunRepository(s).start_run(run_id, dag_id)
            RunRepository(s).finish_run(run_id, state=state, gold_snapshot_id=gold_snapshot_id)
    finally:
        db.dispose()
    with bind_context(run_id=run_id, dag_id=dag_id):
        log.info("run_finished", state=state, gold_snapshot_id=gold_snapshot_id)
    return {"run_id": run_id, "state": state}


def _record(ctx: TaskContext, **metrics: Any) -> None:
    ctx.metrics = metrics  # type: ignore[attr-defined]


def _lineage(
    ctx: TaskContext,
    *,
    output_dataset: str,
    rows_out: int,
    input_dataset: str | None = None,
    rows_in: int | None = None,
) -> None:
    ctx.lineage = {
        "output_dataset": output_dataset,
        "rows_out": rows_out,
        "input_dataset": input_dataset,
        "rows_in": rows_in,
    }


# ---------------------------------------------------------------------------
# akl_ingestion
# ---------------------------------------------------------------------------
def list_connectors(run_id: str, connector_ids: Sequence[str] | None = None) -> list[str]:
    with task_scope(run_id, "akl_ingestion", "load_connector_configs") as ctx:
        from akl.ingestion.service import IngestionService

        svc = IngestionService(ctx.settings, ctx.engine, ctx.db)
        ids = [c.id for c in svc.connector_configs()]
        if connector_ids:
            wanted = set(connector_ids)
            ids = [i for i in ids if i in wanted]
        _record(ctx, rows_out=len(ids))
        return ids


def fetch_connector(run_id: str, connector_id: str, *, map_index: int = -1) -> dict[str, Any]:
    with task_scope(run_id, "akl_ingestion", "fetch_to_bronze", map_index=map_index) as ctx:
        from akl.ingestion.service import IngestionService

        svc = IngestionService(ctx.settings, ctx.engine, ctx.db)
        rep = svc.run_connector(connector_id, run_id=run_id)
        out = {
            "connector_id": connector_id,
            "discovered": rep.discovered,
            "fetched": rep.fetched,
            "deduplicated": rep.deduplicated,
            "failed": rep.failed,
            "deletions": len(rep.deletions),
            "manifest_rows": rep.manifest_rows,
            "failures": rep.failures[:20],
            "duration_s": round(rep.duration_s, 2),
        }
        _record(
            ctx,
            rows_in=rep.discovered,
            rows_out=rep.fetched,
            **{k: v for k, v in out.items() if k not in ("failures", "connector_id")},
        )
        _lineage(
            ctx,
            output_dataset="bronze/manifest",
            rows_out=rep.manifest_rows,
            input_dataset=f"connector/{connector_id}",
            rows_in=rep.discovered,
        )
        return out


def parse_backlog(
    run_id: str, *, limit: int = 500, allow_secret_like: bool = False
) -> dict[str, Any]:
    with task_scope(run_id, "akl_ingestion", "parse_validate_to_silver") as ctx:
        from akl.ingestion.service import IngestionService

        rep = IngestionService(ctx.settings, ctx.engine, ctx.db).parse_backlog(
            run_id=run_id, limit=limit, allow_secret_like=allow_secret_like
        )
        out = {
            "considered": rep.considered,
            "parsed": rep.parsed,
            "skipped": rep.skipped,
            "quarantined": rep.quarantined,
            "duplicates": rep.duplicates,
            "low_quality": rep.low_quality,
            "pii_flagged": rep.pii_flagged,
            "failures": rep.failures[:20],
        }
        _record(
            ctx,
            rows_in=rep.considered,
            rows_out=rep.parsed,
            **{k: v for k, v in out.items() if k != "failures"},
        )
        _lineage(
            ctx,
            output_dataset="silver/documents",
            rows_out=rep.parsed,
            input_dataset="bronze/manifest",
            rows_in=rep.considered,
        )
        return out


def ingestion_gate(
    fetch_reports: Sequence[dict[str, Any]],
    parse_report: dict[str, Any],
    *,
    max_quarantine_ratio: float = 0.25,
) -> dict[str, Any]:
    """PRD §7.8: fail when quarantined/fetched > ratio, or when there was a backlog but nothing parsed."""
    fetched = sum(int(r.get("fetched", 0)) for r in fetch_reports)
    quarantined = int(parse_report.get("quarantined", 0))
    considered = int(parse_report.get("considered", 0))
    parsed = int(parse_report.get("parsed", 0))
    ratio = quarantined / fetched if fetched else 0.0
    result = {
        "fetched": fetched,
        "considered": considered,
        "parsed": parsed,
        "quarantined": quarantined,
        "quarantine_ratio": round(ratio, 4),
        "passed": True,
    }
    if fetched and ratio > max_quarantine_ratio:
        raise GateFailed(f"quarantine ratio {ratio:.2f} > {max_quarantine_ratio}", details=result)
    if considered > 0 and parsed == 0 and quarantined == 0:
        raise GateFailed("backlog present but nothing parsed", details=result)
    return result


# ---------------------------------------------------------------------------
# akl_chunking
# ---------------------------------------------------------------------------
def chunk_run(
    run_id: str,
    *,
    limit: int = 200,
    document_ids: Sequence[str] | None = None,
    refresh_gold: bool = True,
) -> dict[str, Any]:
    with task_scope(run_id, "akl_chunking", "chunk_documents") as ctx:
        from akl.chunking.incremental import ChunkingService

        svc = ChunkingService(ctx.settings, ctx.engine, ctx.db)
        rep = svc.run(
            run_id=run_id,
            document_ids=[uuid.UUID(d) for d in document_ids] if document_ids else None,
            limit=limit,
            refresh_gold=refresh_gold,
        )
        out = {
            "documents_considered": rep.documents_considered,
            "documents_chunked": rep.documents_chunked,
            "documents_unchanged": rep.documents_unchanged,
            "documents_failed": rep.documents_failed,
            "chunks_written": rep.chunks_written,
            "chunks_tombstoned": rep.chunks_tombstoned,
            "unchanged": rep.unchanged,
            "modified": rep.modified,
            "moved": rep.moved,
            "added": rep.added,
            "removed": rep.removed,
            "reparented": rep.reparented,
            "gold_rows_promoted": rep.gold_rows_promoted,
            "gold_snapshot_id": rep.gold_snapshot_id,
            "failures": rep.failures[:20],
        }
        _record(
            ctx,
            rows_in=rep.documents_considered,
            rows_out=rep.chunks_written,
            **{k: v for k, v in out.items() if k not in ("failures",)},
        )
        _lineage(
            ctx,
            output_dataset="silver/chunks",
            rows_out=rep.chunks_written,
            input_dataset="silver/documents",
            rows_in=rep.documents_considered,
        )
        return out


def chunking_gate(report: dict[str, Any], *, max_failed_ratio: float = 0.3) -> dict[str, Any]:
    considered = int(report.get("documents_considered", 0))
    failed = int(report.get("documents_failed", 0))
    ratio = failed / considered if considered else 0.0
    result = {
        "considered": considered,
        "failed": failed,
        "failed_ratio": round(ratio, 4),
        "passed": True,
    }
    if considered and ratio > max_failed_ratio:
        raise GateFailed(f"chunking failure ratio {ratio:.2f} > {max_failed_ratio}", details=result)
    return result


# ---------------------------------------------------------------------------
# akl_embedding
# ---------------------------------------------------------------------------
def warm_model(run_id: str) -> dict[str, Any]:
    with task_scope(run_id, "akl_embedding", "warm_model_check") as ctx:
        from akl.embedding.provider import build_provider

        provider = build_provider(ctx.settings.embedding, ctx.settings.core.models_dir)
        vec = provider.embed_query("warm up")
        out = {
            "model": provider.model_id,
            "embedding_version": provider.embedding_version,
            "dim": int(vec.shape[0]),
        }
        if out["dim"] != ctx.settings.embedding.embed_dim:
            raise GateFailed("embedding dimension mismatch", details=out)
        _record(ctx, **out)
        return out


def embed_run(run_id: str, *, limit: int | None = None) -> dict[str, Any]:
    with task_scope(run_id, "akl_embedding", "embed_backlog") as ctx:
        from akl.embedding.pipeline import EmbeddingPipeline

        pipeline = EmbeddingPipeline(ctx.settings, ctx.engine, ctx.db)
        rep = pipeline.run(run_id=run_id, limit=limit)
        coverage, backlog = pipeline.coverage()
        out = {
            "embedding_version": rep.embedding_version,
            "backlog": rep.backlog,
            "cache_hits": rep.cache_hits,
            "generated": rep.generated,
            "written": rep.written,
            "failed": rep.failed,
            "batches": rep.batches,
            "throughput_cps": rep.throughput_cps,
            "coverage": round(coverage, 4),
            "remaining_backlog": backlog,
            "job_id": str(rep.job_id) if rep.job_id else None,
            "failures": rep.failures[:10],
        }
        _record(
            ctx,
            rows_in=rep.backlog,
            rows_out=rep.written,
            **{k: v for k, v in out.items() if k not in ("failures", "job_id")},
        )
        _lineage(
            ctx,
            output_dataset="gold/chunk_embeddings",
            rows_out=rep.written,
            input_dataset="silver/chunks",
            rows_in=rep.backlog,
        )
        return out


def coverage_gate(report: dict[str, Any], *, min_coverage: float = 0.99) -> dict[str, Any]:
    coverage = float(report.get("coverage", 0.0))
    result = {
        "coverage": coverage,
        "min_coverage": min_coverage,
        "passed": coverage >= min_coverage,
    }
    if coverage < min_coverage:
        raise GateFailed(f"embedding coverage {coverage:.3f} < {min_coverage}", details=result)
    return result


# ---------------------------------------------------------------------------
# akl_qdrant_sync
# ---------------------------------------------------------------------------
def qdrant_health(run_id: str) -> dict[str, Any]:
    with task_scope(run_id, "akl_qdrant_sync", "qdrant_health_sensor") as ctx:
        from akl.embedding.qdrant.schema import make_client

        client = make_client(ctx.settings)
        names = [c.name for c in client.get_collections().collections]
        _record(ctx, collections=len(names))
        return {"ok": True, "collections": names}


def qdrant_sync(run_id: str, *, dry_run: bool = False) -> dict[str, Any]:
    with task_scope(run_id, "akl_qdrant_sync", "reconcile") as ctx:
        from akl.embedding.pipeline import EmbeddingPipeline
        from akl.embedding.qdrant.reconciler import QdrantReconciler
        from akl.embedding.qdrant.schema import make_client

        pipeline = EmbeddingPipeline(ctx.settings, ctx.engine, ctx.db, allow_download=False)
        rec = QdrantReconciler(make_client(ctx.settings), ctx.settings, ctx.engine, pipeline.gold)
        rep = rec.sync(run_id=run_id, dry_run=dry_run)
        out = {
            "collection": rep.collection,
            "embedding_version": rep.embedding_version,
            "gold_points": rep.gold_points,
            "before": rep.qdrant_points_before,
            "to_upsert": rep.to_upsert,
            "to_delete": rep.to_delete,
            "upserted": rep.upserted,
            "deleted": rep.deleted,
            "after": rep.qdrant_points_after,
            "drift": rep.drift,
            "dry_run": dry_run,
        }
        _record(
            ctx,
            rows_in=rep.gold_points,
            rows_out=rep.upserted,
            **{k: v for k, v in out.items() if k not in ("collection", "embedding_version")},
        )
        return out


def bm25_build(run_id: str) -> dict[str, Any]:
    with task_scope(run_id, "akl_qdrant_sync", "rebuild_bm25_index") as ctx:
        from akl.embedding.bm25.builder import build_bm25
        from akl.lakehouse.gold import GoldStore
        from akl.lakehouse.io import LakehouseIO

        io = LakehouseIO(ctx.settings, ctx.engine)
        gold = GoldStore(
            io,
            ctx.engine,
            embedding_version=ctx.settings.embedding.embedding_version,
            view_params={"chunker_version": ctx.settings.chunking.chunker_version},
        )
        rep = build_bm25(ctx.settings, io, gold, version=run_id)
        out = {
            "version": rep.version,
            "documents": rep.documents,
            "terms": rep.terms,
            "prefix": rep.prefix,
        }
        _record(ctx, rows_out=rep.documents, terms=rep.terms)
        _lineage(
            ctx,
            output_dataset="gold/indexes/bm25",
            rows_out=rep.documents,
            input_dataset="gold/retrieval_units",
        )
        return out


def notify_api_reload(
    run_id: str, *, api_url: str | None, timeout_s: float = 10.0
) -> dict[str, Any]:
    """POST /v1/admin/reload-index with a short-lived service token; a missing API is not an error."""
    with task_scope(run_id, "akl_qdrant_sync", "notify_api_reload") as ctx:
        if not api_url:
            return {"skipped": True, "reason": "no api url configured"}
        import httpx

        from akl.security.auth import Authenticator

        token = Authenticator(ctx.settings, None).mint_token(
            "airflow", groups=[], security_levels=["public"], roles=["service"], ttl_s=300
        )
        try:
            resp = httpx.post(
                f"{api_url.rstrip('/')}/v1/admin/reload-index",
                headers={"Authorization": f"Bearer {token}"},
                timeout=timeout_s,
            )
            out = {"skipped": False, "status": resp.status_code, "body": resp.text[:200]}
        except httpx.HTTPError as exc:
            out = {"skipped": True, "reason": f"api unreachable: {exc}"}
        _record(ctx, **{k: v for k, v in out.items() if k != "body"})
        return out


# ---------------------------------------------------------------------------
# akl_maintenance
# ---------------------------------------------------------------------------
def maintenance_task(
    run_id: str,
    operation: str,
    task_id: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Dispatch one maintenance operation by name; ``task_id`` defaults to the operation name."""
    with task_scope(run_id, "akl_maintenance", task_id or operation) as ctx:
        from akl.lakehouse.io import LakehouseIO
        from akl.pipelines import maintenance as m

        io = LakehouseIO(ctx.settings, ctx.engine)
        ops: dict[str, Callable[[], dict[str, Any]]] = {
            "compact_partitions": lambda: _compact(ctx, io, run_id, **kwargs),
            "bronze_retention": lambda: m.bronze_retention(
                io,
                ctx.db,
                days=int(kwargs.get("days", 365)),
                dry_run=bool(kwargs.get("dry_run", False)),
            ),
            "quarantine_retention": lambda: m.quarantine_retention(
                io,
                ctx.db,
                days=int(kwargs.get("days", 90)),
                dry_run=bool(kwargs.get("dry_run", False)),
            ),
            "embedding_cache_eviction": lambda: m.embedding_cache_eviction(
                ctx.db,
                ttl_days=int(
                    kwargs.get("ttl_days", ctx.settings.embedding.embedding_cache_ttl_days)
                ),
            ),
            "conversation_ttl": lambda: m.conversation_ttl(ctx.db),
            "retire_old_embedding_versions": lambda: m.retire_old_embedding_versions(
                io,
                current_version=ctx.settings.embedding.embedding_version,
                days=int(kwargs.get("days", ctx.settings.embedding.embedding_retire_days)),
                dry_run=bool(kwargs.get("dry_run", False)),
            ),
            "compute_corpus_stats": lambda: _stats(ctx, io, run_id),
            "backup_postgres": lambda: m.backup_postgres(ctx.settings, io),
            "qdrant_snapshot": lambda: m.qdrant_snapshot(ctx.settings, io),
            "backup_retention": lambda: m.backup_retention(
                io, days=int(kwargs.get("days", 14)), dry_run=bool(kwargs.get("dry_run", False))
            ),
            "audit_log_retention": lambda: m.audit_log_retention(
                ctx.db,
                days=int(kwargs.get("days", ctx.settings.governance.audit_log_retention_days)),
                dry_run=bool(kwargs.get("dry_run", False)),
            ),
            "vacuum_analyze": lambda: m.vacuum_analyze(ctx.db),
        }
        if operation not in ops:
            raise AKLError(
                f"unknown maintenance operation {operation!r}", details={"available": sorted(ops)}
            )
        out = ops[operation]()
        _record(ctx, **{k: v for k, v in out.items() if isinstance(v, int | float | str | bool)})
        return out


def _compact(ctx: TaskContext, io: Any, run_id: str, **kwargs: Any) -> dict[str, Any]:
    from akl.lakehouse.compaction import compact

    rep = compact(
        io,
        ctx.engine,
        run_id=run_id,
        min_files=int(kwargs.get("min_files", ctx.settings.lakehouse.compact_min_files)),
        small_file_mb=int(
            kwargs.get("small_file_mb", ctx.settings.lakehouse.compact_small_file_mb)
        ),
        dry_run=bool(kwargs.get("dry_run", False)),
    )
    return {
        "planned": len(rep.planned),
        "compacted": rep.compacted,
        "files_merged": rep.files_merged,
        "bytes_before": rep.bytes_before,
        "bytes_after": rep.bytes_after,
        "skipped": rep.skipped,
        "errors": rep.errors[:10],
        "partitions": [f"{p.layer}/{p.dataset}/{p.partition}" for p in rep.planned][:20],
    }


def _stats(ctx: TaskContext, io: Any, run_id: str) -> dict[str, Any]:
    from akl.lakehouse.gold import GoldStore
    from akl.pipelines.maintenance import compute_corpus_stats

    gold = GoldStore(
        io,
        ctx.engine,
        embedding_version=ctx.settings.embedding.embedding_version,
        view_params={"chunker_version": ctx.settings.chunking.chunker_version},
    )
    return compute_corpus_stats(gold, run_id=run_id)
