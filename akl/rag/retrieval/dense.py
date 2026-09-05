"""Dense retrieval via Qdrant (PRD §6.3.1)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qdrant_client.http import models as qm

from akl.embedding.provider import EmbeddingProvider
from akl.embedding.qdrant.reconciler import QdrantReconciler
from akl.rag.retrieval.models import Candidate


class DenseRetriever:
    def __init__(
        self, reconciler: QdrantReconciler, provider: EmbeddingProvider, *, use_alias: bool = True
    ) -> None:
        self._qd = reconciler
        self._provider = provider
        self.use_alias = use_alias

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [self._provider.embed_query(t) for t in texts]

    def search(
        self,
        vectors: Sequence[np.ndarray],
        *,
        k: int,
        query_filter: qm.Filter,
        hnsw_ef: int | None = None,
    ) -> list[Candidate]:
        """Search each query variant; merge by best score; rank by score."""
        best: dict[str, Candidate] = {}
        for vec in vectors:
            for hit in self._qd.search(
                vec, limit=k, query_filter=query_filter, hnsw_ef=hnsw_ef, use_alias=self.use_alias
            ):
                cid = hit["chunk_id"]
                score = float(hit.pop("score"))
                cur = best.get(cid)
                if cur is None or (cur.dense_score or -1) < score:
                    best[cid] = Candidate(chunk_id=cid, payload=hit, dense_score=score)
        ranked = sorted(best.values(), key=lambda c: -(c.dense_score or 0.0))[:k]
        for rank, cand in enumerate(ranked, start=1):
            cand.dense_rank = rank
        return ranked
