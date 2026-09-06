"""Maintenance operations (PRD §7.7, §9.10, §11.10): retention, eviction, stats, backups.

Every function is idempotent and returns a JSON-serialisable dict so it can run from the
Airflow maintenance DAG, the CLI, or tests. Deletions only ever remove objects that the
policy says are expired *and* that nothing current references.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select, text

from akl.config import Settings
from akl.db.models import DocumentVersion, QuarantineItem
from akl.db.repositories.conversations import ConversationRepository
from akl.db.repositories.embedding_cache import EmbeddingCacheRepository
from akl.db.session import Database
from akl.lakehouse.gold import GoldStore
from akl.lakehouse.io import LakehouseIO, Layer


def _cutoff(days: int) -> datetime:
    return datetime.now(UTC) - timedelta(days=days)


def bronze_retention(
    io: LakehouseIO, db: Database, *, days: int, dry_run: bool = False
) -> dict[str, Any]:
    """Delete raw objects older than ``days`` whose sha is not referenced by any document version."""
    cutoff = _cutoff(days)
    with db.session() as s:
        referenced = set(s.scalars(select(DocumentVersion.content_sha256).distinct()))
    candidates = []
    for obj in io.list_objects("bronze/raw/"):
        sha = obj.key.rsplit("sha256=", 1)[-1].split(".", 1)[0]
        if obj.last_modified < cutoff and sha not in referenced:
            candidates.append(obj.key)
    deleted = 0 if dry_run else io.delete_keys(candidates, allow_bronze_raw=True)
    return {
        "scanned": len(referenced),
        "candidates": len(candidates),
        "deleted": deleted,
        "cutoff": cutoff.isoformat(),
        "dry_run": dry_run,
    }


def quarantine_retention(
    io: LakehouseIO, db: Database, *, days: int, dry_run: bool = False
) -> dict[str, Any]:
    cutoff = _cutoff(days)
    keys = [
        o.key
        for o in io.list_objects("quarantine/")
        if o.last_modified < cutoff and not o.key.endswith(".keep")
    ]
    deleted = 0 if dry_run else io.delete_keys(keys)
    rows = 0
    if not dry_run:
        with db.session() as s:
            result = s.execute(
                text("DELETE FROM quarantine_items WHERE created_at < :cutoff"), {"cutoff": cutoff}
            )
            rows = int(getattr(result, "rowcount", 0) or 0)
    else:
        with db.session() as s:
            rows = int(
                s.scalar(
                    select(text("count(*)"))
                    .select_from(QuarantineItem)
                    .where(QuarantineItem.created_at < cutoff)
                )
                or 0
            )
    return {
        "objects_deleted": deleted,
        "rows_deleted": rows,
        "cutoff": cutoff.isoformat(),
        "dry_run": dry_run,
    }


def audit_log_retention(db: Database, *, days: int, dry_run: bool = False) -> dict[str, Any]:
    """Delete ``audit_log`` rows older than ``days`` (PRD §9.5: audit trail is retained, not forever)."""
    cutoff = _cutoff(days)
    with db.session() as s:
        if dry_run:
            rows = int(
                s.scalar(
                    text("SELECT count(*) FROM audit_log WHERE ts < :cutoff").bindparams(
                        cutoff=cutoff
                    )
                )
                or 0
            )
        else:
            result = s.execute(text("DELETE FROM audit_log WHERE ts < :cutoff"), {"cutoff": cutoff})
            rows = int(getattr(result, "rowcount", 0) or 0)
    return {"rows_deleted": rows, "cutoff": cutoff.isoformat(), "dry_run": dry_run}


def embedding_cache_eviction(db: Database, *, ttl_days: int) -> dict[str, Any]:
    with db.session() as s:
        repo = EmbeddingCacheRepository(s)
        before = repo.count()
        evicted = repo.evict_stale(ttl_days=ttl_days)
    return {"before": before, "evicted": evicted, "ttl_days": ttl_days}


def conversation_ttl(db: Database) -> dict[str, Any]:
    with db.session() as s:
        purged = ConversationRepository(s).purge_expired()
    return {"purged": purged}


def retire_old_embedding_versions(
    io: LakehouseIO, *, current_version: str, days: int, dry_run: bool = False
) -> dict[str, Any]:
    """Drop ``gold/chunk_embeddings/embedding_version=<v>/`` partitions for retired versions older than ``days``."""
    cutoff = _cutoff(days)
    by_version: dict[str, list[Any]] = {}
    for obj in io.list_objects("gold/chunk_embeddings/embedding_version="):
        version = obj.key.split("embedding_version=", 1)[1].split("/", 1)[0]
        by_version.setdefault(version, []).append(obj)
    retired: list[str] = []
    deleted = 0
    for version, objs in by_version.items():
        if version == current_version or not objs:
            continue
        if max(o.last_modified for o in objs) < cutoff:
            retired.append(version)
            if not dry_run:
                deleted += io.delete_keys([o.key for o in objs])
    return {
        "versions_seen": sorted(by_version),
        "retired": retired,
        "objects_deleted": deleted,
        "dry_run": dry_run,
    }


def compute_corpus_stats(gold: GoldStore, *, run_id: str) -> dict[str, Any]:
    rows = gold.compute_stats(run_id=run_id)
    return {"rows": len(rows), "metrics": sorted({r["metric"] for r in rows})}


def backup_postgres(
    settings: Settings, io: LakehouseIO, *, database: str | None = None
) -> dict[str, Any]:
    """``pg_dump -Fc`` of the akl database into ``backups/postgres/``; skipped when pg_dump is unavailable."""
    binary = shutil.which("pg_dump")
    if binary is None:
        return {"skipped": True, "reason": "pg_dump not installed"}
    db = settings.db
    name = database or db.name
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    key = f"backups/postgres/{name}-{stamp}.dump"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dump"
        cmd = [
            binary,
            "-Fc",
            "-h",
            db.host,
            "-p",
            str(db.port),
            "-U",
            db.user,
            "-d",
            name,
            "-f",
            str(path),
        ]
        env = {"PGPASSWORD": db.password.get_secret_value(), "PATH": "/usr/bin:/bin:/usr/local/bin"}
        proc = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=1800, env=env
        )
        if proc.returncode != 0:
            return {"skipped": False, "ok": False, "error": proc.stderr[-500:]}
        data = path.read_bytes()
    io.put_object(key, data, content_type="application/octet-stream")
    return {"skipped": False, "ok": True, "key": key, "bytes": len(data)}


def qdrant_snapshot(
    settings: Settings, io: LakehouseIO, *, collection: str | None = None, timeout_s: float = 300.0
) -> dict[str, Any]:
    """Create a collection snapshot and copy it to ``backups/qdrant/``."""
    coll = collection or settings.qdrant.collection
    base = settings.qdrant.url.rstrip("/")
    headers = (
        {"api-key": settings.qdrant.api_key.get_secret_value()} if settings.qdrant.api_key else {}
    )
    with httpx.Client(base_url=base, headers=headers, timeout=timeout_s) as client:
        created = client.post(f"/collections/{coll}/snapshots")
        created.raise_for_status()
        name = str(created.json()["result"]["name"])
        blob = client.get(f"/collections/{coll}/snapshots/{name}")
        blob.raise_for_status()
        key = f"backups/qdrant/{coll}/{name}"
        io.put_object(key, blob.content, content_type="application/octet-stream")
        client.delete(f"/collections/{coll}/snapshots/{name}")  # server-side copy no longer needed
    return {"collection": coll, "snapshot": name, "key": key, "bytes": len(blob.content)}


def backup_retention(io: LakehouseIO, *, days: int, dry_run: bool = False) -> dict[str, Any]:
    cutoff = _cutoff(days)
    keys = [
        o.key
        for o in io.list_objects("backups/")
        if o.last_modified < cutoff and not o.key.endswith(".keep")
    ]
    return {
        "deleted": 0 if dry_run else io.delete_keys(keys),
        "candidates": len(keys),
        "dry_run": dry_run,
    }


def vacuum_analyze(
    db: Database,
    tables: Sequence[str] = (
        "documents",
        "document_versions",
        "chunks",
        "embedding_cache",
        "retrieval_traces",
        "messages",
    ),
) -> dict[str, Any]:
    done: list[str] = []
    with db.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        for table in tables:
            conn.execute(text(f"VACUUM ANALYZE {table}"))  # noqa: S608 - fixed table names
            done.append(table)
    return {"tables": done}


def compaction_layers() -> list[str]:
    return [layer.value for layer in (Layer.BRONZE, Layer.SILVER, Layer.GOLD)]
