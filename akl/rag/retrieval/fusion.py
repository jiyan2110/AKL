"""Reciprocal Rank Fusion with quality and soft-filter adjustments (PRD §6.3.3, ADR-005)."""

from __future__ import annotations

from collections.abc import Sequence

from akl.rag.retrieval.models import Candidate


def rrf_fuse(
    dense: Sequence[Candidate],
    sparse: Sequence[Candidate],
    *,
    k: int = 60,
    fused_k: int = 40,
    weight_dense: float = 1.0,
    weight_sparse: float = 1.0,
    marginal_penalty: float = 0.9,
    soft_bonus: float = 1.05,
) -> list[Candidate]:
    fused: dict[str, Candidate] = {}
    for cand in dense:
        merged = fused.setdefault(cand.chunk_id, Candidate(cand.chunk_id, cand.payload))
        merged.dense_score, merged.dense_rank = cand.dense_score, cand.dense_rank
        merged.soft_match = merged.soft_match and cand.soft_match
        merged.rrf_score += weight_dense / (k + (cand.dense_rank or 1))
    for cand in sparse:
        merged = fused.setdefault(cand.chunk_id, Candidate(cand.chunk_id, cand.payload))
        merged.sparse_score, merged.sparse_rank = cand.sparse_score, cand.sparse_rank
        merged.soft_match = merged.soft_match and cand.soft_match
        if not merged.payload:
            merged.payload = cand.payload
        merged.rrf_score += weight_sparse / (k + (cand.sparse_rank or 1))
    for cand in fused.values():
        flags = cand.payload.get("quality_flags") or []
        if "marginal" in flags:
            cand.rrf_score *= marginal_penalty
            cand.flags.append("marginal")
        if cand.soft_match and soft_bonus != 1.0:
            cand.rrf_score *= soft_bonus
    return sorted(fused.values(), key=lambda c: -c.rrf_score)[:fused_k]
