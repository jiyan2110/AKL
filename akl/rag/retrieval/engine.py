"""HybridRetriever: dense ∥ sparse → RRF → rerank → confidence gate (PRD §6.3, §6.14)."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from akl.config import RetrievalSettings
from akl.embedding.provider import EmbeddingModelError
from akl.errors import AKLError
from akl.rag.query.filters import allowed_by_filters, to_qdrant_filter
from akl.rag.query.processor import ProcessedQuery
from akl.rag.retrieval.dense import DenseRetriever
from akl.rag.retrieval.fusion import rrf_fuse
from akl.rag.retrieval.models import Candidate
from akl.rag.retrieval.rerank import LexicalReranker, Reranker
from akl.rag.retrieval.sparse import SparseRetriever


class RetrievalUnavailableError(AKLError):
    """Both retrieval backends failed (AKL-E6012)."""

    code = "AKL-E6012"
    http_status = 503
    retryable = True


@dataclass
class RetrievalResult:
    candidates: list[Candidate]
    confidence: float
    sufficient: bool
    reason: str | None
    dense_ids: list[str]
    sparse_ids: list[str]
    fused_ids: list[str]
    flags: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    reranker: str | None = None

    def top(self, k: int) -> list[Candidate]:
        return self.candidates[:k]

    def trace(self) -> dict[str, Any]:
        return {
            "dense_ids": self.dense_ids,
            "sparse_ids": self.sparse_ids,
            "fused_ids": self.fused_ids,
            "reranked": [{"chunk_id": c.chunk_id, **c.scores()} for c in self.candidates[:20]],
            "confidence": self.confidence,
            "sufficient": self.sufficient,
            "reason": self.reason,
            "flags": self.flags,
            "reranker": self.reranker,
            "timings_ms": self.timings_ms,
        }


class HybridRetriever:
    def __init__(
        self,
        settings: RetrievalSettings,
        *,
        dense: DenseRetriever | None,
        sparse: SparseRetriever | None,
        reranker: Reranker | None,
    ) -> None:
        self.settings = settings
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self._fallback = LexicalReranker()

    # -- public ----------------------------------------------------------------------------
    def retrieve(
        self, query: ProcessedQuery, *, mode: str = "hybrid", rerank: bool | None = None
    ) -> RetrievalResult:
        if self.dense is None and self.sparse is None:
            raise RetrievalUnavailableError("no retrieval backend available")
        timings: dict[str, float] = {}
        flags: list[str] = []
        s = self.settings
        want_dense = mode in ("hybrid", "dense") and self.dense is not None
        want_sparse = mode in ("hybrid", "sparse") and self.sparse is not None
        if mode == "dense" and self.dense is None:
            raise RetrievalUnavailableError("dense backend unavailable")
        if mode == "sparse" and self.sparse is None:
            raise RetrievalUnavailableError("sparse backend unavailable")

        soft_active = not query.soft_filters.is_empty()
        dense_res, sparse_res = self._parallel(
            query, want_dense, want_sparse, soft_active, timings, flags
        )
        # two-pass soft filters (PRD §6.2.5): if the filtered pass is thin, rerun unfiltered and down-weight
        if soft_active and len(dense_res) + len(sparse_res) < s.rag_top_k:
            t0 = time.perf_counter()
            d2, s2 = self._parallel(query, want_dense, want_sparse, False, {}, [])
            seen = {c.chunk_id for c in [*dense_res, *sparse_res]}
            for cand in [*d2, *s2]:
                if cand.chunk_id not in seen:
                    cand.soft_match = False
            dense_res = dense_res + [
                c for c in d2 if c.chunk_id not in {x.chunk_id for x in dense_res}
            ]
            sparse_res = sparse_res + [
                c for c in s2 if c.chunk_id not in {x.chunk_id for x in sparse_res}
            ]
            for rank, c in enumerate(dense_res, start=1):
                c.dense_rank = rank
            for rank, c in enumerate(sparse_res, start=1):
                c.sparse_rank = rank
            flags.append("soft_filter_relaxed")
            timings["soft_relax"] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        fused = rrf_fuse(
            dense_res,
            sparse_res,
            k=s.rrf_k,
            fused_k=s.retrieval_fused_k,
            weight_sparse=float(query.profile.get("sparse_weight", 1.0)),
            marginal_penalty=s.rag_marginal_penalty,
            soft_bonus=1.05 if soft_active else 1.0,
        )
        for cand in fused:
            if not cand.soft_match:
                cand.rrf_score *= s.rag_soft_filter_penalty
        boost_types = tuple(query.profile.get("boost_chunk_types", ()))
        if boost_types:
            for cand in fused:
                if cand.payload.get("chunk_type") in boost_types:
                    cand.rrf_score *= 1.05
        fused.sort(key=lambda c: -c.rrf_score)
        timings["fusion"] = round((time.perf_counter() - t0) * 1000, 1)

        do_rerank = s.rerank_enabled if rerank is None else rerank
        reranker_name: str | None = None
        if do_rerank and fused:
            t0 = time.perf_counter()
            reranker = self.reranker or self._fallback
            try:
                fused = reranker.rerank(query.dense_text, fused, top_n=s.rerank_top_n)
                reranker_name = reranker.name
            except (EmbeddingModelError, AKLError, RuntimeError, OSError) as exc:
                flags.append("reranker_fallback")
                fused = self._fallback.rerank(query.dense_text, fused, top_n=s.rerank_top_n)
                reranker_name = f"lexical(fallback:{type(exc).__name__})"
            timings["rerank"] = round((time.perf_counter() - t0) * 1000, 1)

        confidence = float(fused[0].final_score) if fused else 0.0
        # Secondary guard (PRD §6.3.5): in the borderline band [min_confidence, strong_confidence)
        # require rag_min_candidates candidates above the evidence floor; a single candidate at or
        # above strong_confidence is sufficient on its own.
        strong = sum(1 for c in fused if c.final_score >= 0.20) if do_rerank else len(fused)
        borderline = confidence < s.rag_strong_confidence
        if not fused:
            sufficient, reason = False, "no_candidates"
        elif confidence < s.rag_min_confidence:
            sufficient, reason = False, "low_confidence"
        elif borderline and strong < s.rag_min_candidates:
            sufficient, reason = False, "insufficient_candidates"
        else:
            sufficient, reason = True, None
        return RetrievalResult(
            candidates=fused,
            confidence=round(confidence, 4),
            sufficient=sufficient,
            reason=reason,
            dense_ids=[c.chunk_id for c in dense_res],
            sparse_ids=[c.chunk_id for c in sparse_res],
            fused_ids=[c.chunk_id for c in fused],
            flags=flags,
            timings_ms=timings,
            reranker=reranker_name,
        )

    # -- internals ------------------------------------------------------------------------------
    def _parallel(
        self,
        query: ProcessedQuery,
        want_dense: bool,
        want_sparse: bool,
        use_soft: bool,
        timings: dict[str, float],
        flags: list[str],
    ) -> tuple[list[Candidate], list[Candidate]]:
        s = self.settings
        soft = query.soft_filters if use_soft else None

        def allowed(row: dict[str, Any]) -> bool:
            return allowed_by_filters(row, query.principal, query.hard_filters, soft)

        def run_dense() -> list[Candidate]:
            assert self.dense is not None
            t0 = time.perf_counter()
            vectors = self.dense.embed(query.dense_variants)
            timings["embed_query"] = round((time.perf_counter() - t0) * 1000, 1)
            t1 = time.perf_counter()
            res = self.dense.search(
                vectors,
                k=s.retrieval_dense_k,
                query_filter=to_qdrant_filter(query.principal, query.hard_filters, soft),
                hnsw_ef=s.qdrant_hnsw_ef,
            )
            timings["dense"] = round((time.perf_counter() - t1) * 1000, 1)
            return res

        def run_sparse() -> list[Candidate]:
            assert self.sparse is not None
            t0 = time.perf_counter()
            res = self.sparse.search(
                query.sparse_text,
                k=s.retrieval_sparse_k,
                allowed=allowed,
                exact_terms=query.exact_terms,
            )
            timings["sparse"] = round((time.perf_counter() - t0) * 1000, 1)
            return res

        dense_res: list[Candidate] = []
        sparse_res: list[Candidate] = []
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            fd = pool.submit(run_dense) if want_dense else None
            fs = pool.submit(run_sparse) if want_sparse else None
            if fd is not None:
                try:
                    dense_res = fd.result()
                except Exception as exc:
                    flags.append("dense_unavailable")
                    errors.append(f"dense: {exc}")
            if fs is not None:
                try:
                    sparse_res = fs.result()
                except Exception as exc:
                    flags.append("sparse_failed")
                    errors.append(f"sparse: {type(exc).__name__}: {exc}")
        if (
            (want_dense or want_sparse)
            and not dense_res
            and not sparse_res
            and len(errors) == int(want_dense) + int(want_sparse)
            and errors
        ):
            raise RetrievalUnavailableError(
                "all retrieval backends failed", details={"errors": errors}
            )
        return dense_res, sparse_res
