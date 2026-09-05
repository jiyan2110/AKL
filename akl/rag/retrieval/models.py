"""Candidate model shared by dense, sparse, fusion and reranking stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    chunk_id: str
    payload: dict[str, Any]
    dense_score: float | None = None
    sparse_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    soft_match: bool = True
    flags: list[str] = field(default_factory=list)

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.rrf_score

    @property
    def text(self) -> str:
        return str(self.payload.get("text") or "")

    @property
    def document_id(self) -> str:
        return str(self.payload.get("document_id") or "")

    def scores(self) -> dict[str, float | None]:
        return {
            "dense": self.dense_score,
            "sparse": self.sparse_score,
            "rrf": round(self.rrf_score, 6),
            "rerank": self.rerank_score,
        }
