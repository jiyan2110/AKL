"""EmbeddingCacheRepository — vectors keyed by (embedded_text_sha256, model) (PRD §5.4, ADR-004)."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from akl.db.models import EmbeddingCache, EmbeddingJob
from akl.db.repositories import Repository
from akl.embedding.provider import bytes_to_vector, vector_to_bytes


class EmbeddingCacheRepository(Repository):
    def lookup(
        self, shas: Sequence[str], *, model_id: str, model_version: str, dim: int
    ) -> dict[str, np.ndarray]:
        """Return vectors for the cache hits and bump their hit counters."""
        if not shas:
            return {}
        stmt = select(EmbeddingCache).where(
            EmbeddingCache.embedded_text_sha256.in_(list(shas)),
            EmbeddingCache.model_id == model_id,
            EmbeddingCache.model_version == model_version,
        )
        rows = list(self.session.scalars(stmt))
        hits = {
            r.embedded_text_sha256: bytes_to_vector(r.vector, dim) for r in rows if r.dim == dim
        }
        if hits:
            self.session.execute(
                update(EmbeddingCache)
                .where(
                    EmbeddingCache.embedded_text_sha256.in_(list(hits)),
                    EmbeddingCache.model_id == model_id,
                    EmbeddingCache.model_version == model_version,
                )
                .values(hit_count=EmbeddingCache.hit_count + 1, last_hit_at=func.now())
            )
        return hits

    def store(
        self,
        items: Iterable[tuple[str, np.ndarray]],
        *,
        model_id: str,
        model_version: str,
        dim: int,
    ) -> int:
        rows = [
            {
                "embedded_text_sha256": sha,
                "model_id": model_id,
                "model_version": model_version,
                "dim": dim,
                "vector": vector_to_bytes(vec),
            }
            for sha, vec in items
        ]
        if not rows:
            return 0
        stmt = (
            pg_insert(EmbeddingCache)
            .values(rows)
            .on_conflict_do_nothing(
                index_elements=[
                    EmbeddingCache.embedded_text_sha256,
                    EmbeddingCache.model_id,
                    EmbeddingCache.model_version,
                ]
            )
        )
        self.session.execute(stmt)
        return len(rows)

    def evict_stale(self, *, ttl_days: int) -> int:
        cutoff = datetime.now(UTC) - timedelta(days=ttl_days)
        result = self.session.execute(
            delete(EmbeddingCache).where(
                func.coalesce(EmbeddingCache.last_hit_at, EmbeddingCache.created_at) < cutoff
            )
        )
        return int(getattr(result, "rowcount", 0) or 0)

    def count(self) -> int:
        return int(self.session.scalar(select(func.count()).select_from(EmbeddingCache)) or 0)

    # -- jobs -------------------------------------------------------------------------
    def record_job(self, **fields: Any) -> uuid.UUID:
        job = EmbeddingJob(job_id=uuid.uuid4(), **fields)
        self.session.add(job)
        return job.job_id
