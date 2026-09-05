"""Sparse retrieval via the BM25 artefact (PRD §6.3.2)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from akl.embedding.bm25.index import Bm25Index
from akl.rag.retrieval.models import Candidate


class SparseRetriever:
    def __init__(self, index: Bm25Index) -> None:
        self.index = index

    def search(
        self,
        query: str,
        *,
        k: int,
        allowed: Callable[[dict[str, Any]], bool],
        exact_terms: Sequence[str] = (),
    ) -> list[Candidate]:
        hits = self.index.search(query, k=k, allowed=allowed, exact_terms=exact_terms)
        out: list[Candidate] = []
        for rank, hit in enumerate(hits, start=1):
            out.append(
                Candidate(
                    chunk_id=hit.chunk_id,
                    payload=dict(hit.payload),
                    sparse_score=hit.score,
                    sparse_rank=rank,
                )
            )
        return out
