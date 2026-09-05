"""Gold → Qdrant reconciliation (PRD §5.14, ADR-001) and dense search helper."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from akl.config import Settings
from akl.embedding.qdrant.schema import PAYLOAD_FIELDS, QdrantSchema, QdrantUnavailableError
from akl.errors import AKLError
from akl.lakehouse.engine import DuckDBEngine
from akl.lakehouse.gold import GoldStore
from akl.lakehouse.schemas.gold import CHUNK_EMBEDDINGS


class QdrantDriftError(AKLError):
    """Qdrant point count differs from Gold after sync (AKL-E5020)."""

    code = "AKL-E5020"
    retryable = True


@dataclass
class SyncReport:
    run_id: str
    collection: str
    embedding_version: str
    gold_points: int = 0
    qdrant_points_before: int = 0
    to_upsert: int = 0
    to_delete: int = 0
    upserted: int = 0
    deleted: int = 0
    qdrant_points_after: int = 0
    drift: int = 0
    dry_run: bool = False
    upsert_ids: list[str] = field(default_factory=list)
    delete_ids: list[str] = field(default_factory=list)


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in PAYLOAD_FIELDS:
        value = row.get(key)
        if isinstance(value, datetime):
            value = int(value.timestamp())
        elif isinstance(value, uuid.UUID):
            value = str(value)
        payload[key] = value
    payload["chunk_id"] = str(row["chunk_id"])
    payload["untrusted"] = (
        row.get("source_type") == "html" and row.get("security_level") == "public"
    )
    return payload


class QdrantReconciler:
    def __init__(
        self,
        client: QdrantClient,
        settings: Settings,
        engine: DuckDBEngine,
        gold: GoldStore,
        *,
        collection: str | None = None,
        manage_alias: bool = True,
    ) -> None:
        self.client = client
        self.settings = settings
        self.engine = engine
        self.gold = gold
        self.embedding_version = gold.embedding_version
        self.collection = collection or settings.qdrant.collection
        self.schema = QdrantSchema(
            client,
            settings,
            dim=gold.embedding_dim,
            collection=self.collection,
            manage_alias=manage_alias,
        )

    # -- gold side ---------------------------------------------------------------------
    def gold_index(self) -> dict[str, str]:
        """``chunk_id → embedded_text_sha256`` for active units that have a vector of this version."""
        self.gold.ensure_views(refresh=True)
        rows = self.engine.execute(
            "SELECT chunk_id, embedded_text_sha256 FROM v_embedding_coverage WHERE has_embedding AND NOT stale_embedding"
        ).to_pylist()
        return {str(r["chunk_id"]): str(r["embedded_text_sha256"]) for r in rows}

    def _gold_points(self, chunk_ids: list[str]) -> list[qm.PointStruct]:
        if not chunk_ids:
            return []
        ids_sql = ", ".join(f"'{c}'" for c in chunk_ids)
        emb_src = self.gold.dataset_source("chunk_embeddings", CHUNK_EMBEDDINGS)
        rows = self.engine.execute(
            f"""
            WITH emb AS (
                SELECT e.chunk_id, e.vector
                FROM {emb_src} AS e
                WHERE e.embedding_version = '{self.embedding_version}' AND e.chunk_id IN ({ids_sql})
                QUALIFY row_number() OVER (PARTITION BY e.chunk_id ORDER BY e.embedded_at DESC) = 1
            )
            SELECT u.*, emb.vector AS _vector
            FROM v_gold_active_units AS u JOIN emb ON emb.chunk_id = u.chunk_id
            """  # noqa: S608 - uuids only
        ).to_pylist()
        points: list[qm.PointStruct] = []
        for row in rows:
            vector = np.asarray(row.pop("_vector"), dtype=np.float32)
            payload = _payload(row)
            payload["embedding_version"] = self.embedding_version
            points.append(
                qm.PointStruct(id=str(row["chunk_id"]), vector=vector.tolist(), payload=payload)
            )
        return points

    # -- qdrant side -------------------------------------------------------------------
    def qdrant_index(self) -> dict[str, tuple[str, str]]:
        """``point_id → (embedded_text_sha256, embedding_version)`` via paginated scroll."""
        out: dict[str, tuple[str, str]] = {}
        if not self.client.collection_exists(self.collection):
            return out
        offset: Any = None
        try:
            while True:
                points, offset = self.client.scroll(
                    self.collection,
                    limit=self.settings.qdrant.scroll_page,
                    offset=offset,
                    with_payload=["embedded_text_sha256", "embedding_version"],
                    with_vectors=False,
                )
                for p in points:
                    payload = p.payload or {}
                    out[str(p.id)] = (
                        str(payload.get("embedded_text_sha256", "")),
                        str(payload.get("embedding_version", "")),
                    )
                if offset is None:
                    break
        except Exception as exc:
            raise QdrantUnavailableError("scroll failed", details={"error": str(exc)}) from exc
        return out

    # -- reconcile -----------------------------------------------------------------------
    def sync(self, *, run_id: str, dry_run: bool = False, verify: bool = True) -> SyncReport:
        report = SyncReport(
            run_id=run_id,
            collection=self.collection,
            embedding_version=self.embedding_version,
            dry_run=dry_run,
        )
        status = self.schema.ensure()
        report.qdrant_points_before = status.points
        gold = self.gold_index()
        qd = self.qdrant_index()
        report.gold_points = len(gold)
        report.upsert_ids = sorted(
            cid
            for cid, sha in gold.items()
            if cid not in qd or qd[cid] != (sha, self.embedding_version)
        )
        report.delete_ids = sorted(pid for pid in qd if pid not in gold)
        report.to_upsert, report.to_delete = len(report.upsert_ids), len(report.delete_ids)
        if dry_run:
            return report
        try:
            ub, dbatch = self.settings.qdrant.upsert_batch, self.settings.qdrant.delete_batch
            for i in range(0, len(report.upsert_ids), ub):
                points = self._gold_points(report.upsert_ids[i : i + ub])
                if points:
                    self.client.upsert(self.collection, points=points, wait=True)
                    report.upserted += len(points)
            for i in range(0, len(report.delete_ids), dbatch):
                ids = report.delete_ids[i : i + dbatch]
                self.client.delete(
                    self.collection, points_selector=qm.PointIdsList(points=ids), wait=True
                )
                report.deleted += len(ids)
            report.qdrant_points_after = int(self.client.count(self.collection, exact=True).count)
        except Exception as exc:
            raise QdrantUnavailableError("sync failed", details={"error": str(exc)}) from exc
        report.drift = report.qdrant_points_after - report.gold_points
        if verify and report.drift != 0:
            raise QdrantDriftError(
                f"drift after sync: qdrant={report.qdrant_points_after} gold={report.gold_points}",
                details={"drift": report.drift},
            )
        return report

    def delete_documents(self, document_ids: list[str]) -> None:
        """Remove every point of the given documents (used by deletion flows and test cleanup)."""
        if not document_ids or not self.client.collection_exists(self.collection):
            return
        self.client.delete(
            self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="document_id", match=qm.MatchAny(any=document_ids))]
                )
            ),
            wait=True,
        )

    # -- search (dense path; hybrid retrieval arrives in M26) ----------------------------------
    def search(
        self,
        vector: np.ndarray,
        *,
        limit: int = 10,
        query_filter: qm.Filter | None = None,
        hnsw_ef: int | None = None,
        use_alias: bool = True,
    ) -> list[dict[str, Any]]:
        target = self.settings.qdrant.collection_alias if use_alias else self.collection
        try:
            result = self.client.query_points(
                target,
                query=vector.tolist(),
                limit=limit,
                query_filter=query_filter,
                search_params=qm.SearchParams(
                    hnsw_ef=hnsw_ef or self.settings.retrieval.qdrant_hnsw_ef
                ),
                with_payload=True,
            )
        except Exception as exc:
            raise QdrantUnavailableError("search failed", details={"error": str(exc)}) from exc
        return [
            {"chunk_id": str(p.id), "score": float(p.score), **(p.payload or {})}
            for p in result.points
        ]
