"""RAGService (PRD §6.15): wires query processing, hybrid retrieval, context and citations.

Generation (LLM provider, prompt, streaming, memory) arrives with the Chat API milestones;
until then ``answer()`` uses the extractive mode (ADR-010), which is also the permanent
fallback when no LLM is configured.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from akl.chunking.tokenizer import TokenCounter
from akl.config import Settings
from akl.db.repositories.traces import RetrievalTraceRepository
from akl.db.session import Database
from akl.embedding.bm25.index import Bm25Index, Bm25IndexError
from akl.embedding.provider import EmbeddingProvider, build_provider
from akl.embedding.qdrant.reconciler import QdrantReconciler
from akl.embedding.qdrant.schema import make_client
from akl.errors import AKLError
from akl.lakehouse.engine import DuckDBEngine
from akl.lakehouse.gold import GoldStore
from akl.lakehouse.io import LakehouseIO
from akl.rag.citations import CitedAnswer, extractive_answer
from akl.rag.context_builder import BuiltContext, ContextBuilder
from akl.rag.query.filters import MetadataFilters
from akl.rag.query.processor import ProcessedQuery, QueryProcessor
from akl.rag.query.spell import SpellCorrector
from akl.rag.retrieval.dense import DenseRetriever
from akl.rag.retrieval.engine import HybridRetriever, RetrievalResult
from akl.rag.retrieval.rerank import Reranker, build_reranker
from akl.rag.retrieval.sparse import SparseRetriever
from akl.security.principal import Principal


@dataclass
class SearchResponse:
    request_id: str
    trace_id: str
    query: ProcessedQuery
    retrieval: RetrievalResult
    results: list[dict[str, Any]]
    timings_ms: dict[str, float]
    gold_snapshot_id: str | None


@dataclass
class AnswerResponse:
    request_id: str
    trace_id: str
    mode: str
    answer: str | None
    citations: list[dict[str, Any]]
    confidence: float
    reason: str | None
    flags: list[str]
    context: BuiltContext | None
    retrieval: RetrievalResult
    query: ProcessedQuery
    timings_ms: dict[str, float] = field(default_factory=dict)


class RAGService:
    def __init__(
        self,
        settings: Settings,
        engine: DuckDBEngine,
        db: Database | None,
        *,
        provider: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        bm25: Bm25Index | None = None,
        qdrant_collection: str | None = None,
        manage_alias: bool = True,
        qdrant_use_alias: bool = True,
        allow_download: bool = True,
        use_dense: bool = True,
        use_sparse: bool = True,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.db = db
        self.io = LakehouseIO(settings, engine)
        self.provider = provider or build_provider(
            settings.embedding, settings.core.models_dir, allow_download=allow_download
        )
        self.gold = GoldStore(
            self.io,
            engine,
            embedding_version=self.provider.embedding_version,
            embedding_dim=self.provider.dim,
            view_params={"chunker_version": settings.chunking.chunker_version},
        )
        self.flags: list[str] = []
        self.qdrant: QdrantReconciler | None = None
        dense: DenseRetriever | None = None
        if use_dense:
            try:
                self.qdrant = QdrantReconciler(
                    make_client(settings),
                    settings,
                    engine,
                    self.gold,
                    collection=qdrant_collection,
                    manage_alias=manage_alias,
                )
                dense = DenseRetriever(self.qdrant, self.provider, use_alias=qdrant_use_alias)
            except AKLError:
                self.flags.append("dense_unavailable")
        self.bm25: Bm25Index | None = bm25
        sparse: SparseRetriever | None = None
        if use_sparse:
            if self.bm25 is None:
                try:
                    self.bm25 = Bm25Index.load(
                        self.io, k1=settings.retrieval.bm25_k1, b=settings.retrieval.bm25_b
                    )
                except Bm25IndexError as exc:
                    self.flags.append("sparse_unavailable")
                    self.sparse_error = f"{exc.message} {exc.details}"
            if self.bm25 is not None:
                sparse = SparseRetriever(self.bm25)
        self.reranker = (
            reranker
            if reranker is not None
            else (
                build_reranker(
                    settings.retrieval, settings.core.models_dir, allow_download=allow_download
                )
                if settings.retrieval.rerank_enabled
                else None
            )
        )
        self.retriever = HybridRetriever(
            settings.retrieval, dense=dense, sparse=sparse, reranker=self.reranker
        )
        spell = (
            SpellCorrector.from_texts(p.get("text") or "" for p in self.bm25.payloads)
            if self.bm25 is not None
            else None
        )
        repos = (
            sorted({str(p["repo"]) for p in self.bm25.payloads if p.get("repo")})
            if self.bm25 is not None
            else []
        )
        self.processor = QueryProcessor(settings.retrieval, spell=spell, known_repos=repos)
        self.counter = TokenCounter(models_dir=settings.core.models_dir)
        self.context_builder = ContextBuilder(
            self.counter.count,
            budget_tokens=settings.retrieval.rag_context_tokens,
            top_k=settings.retrieval.rag_top_k,
            dedupe_jaccard=settings.retrieval.rag_dedupe_jaccard,
        )

    # -- public -------------------------------------------------------------------------------
    def search(
        self,
        text: str,
        principal: Principal,
        *,
        mode: str = "hybrid",
        k: int | None = None,
        filters: MetadataFilters | None = None,
        rerank: bool | None = None,
        include_text: bool = True,
        request_id: str | None = None,
        persist_trace: bool = True,
    ) -> SearchResponse:
        t0 = time.perf_counter()
        request_id = request_id or uuid.uuid4().hex
        trace_id = uuid.uuid4().hex
        processed = self.processor.process(text, principal, filters=filters)
        t_q = time.perf_counter()
        retrieval = self.retriever.retrieve(processed, mode=mode, rerank=rerank)
        top = retrieval.top(k or self.settings.retrieval.rag_top_k)
        results = []
        for rank, cand in enumerate(top, start=1):
            p = cand.payload
            item: dict[str, Any] = {
                "rank": rank,
                "chunk_id": cand.chunk_id,
                "lineage_id": p.get("lineage_id"),
                "document_id": p.get("document_id"),
                "title": p.get("title"),
                "source_type": p.get("source_type"),
                "heading_breadcrumb": p.get("heading_breadcrumb"),
                "chunk_type": p.get("chunk_type"),
                "locator": _locator(p),
                "url": p.get("source_uri") or p.get("canonical_source_uri"),
                "scores": cand.scores(),
            }
            if include_text:
                item["text"] = cand.text
            results.append(item)
        timings = {
            "query_processing": round((t_q - t0) * 1000, 1),
            **retrieval.timings_ms,
            "total": round((time.perf_counter() - t0) * 1000, 1),
        }
        snapshot = str(top[0].payload.get("gold_snapshot_id")) if top else None
        if persist_trace:
            self._persist(trace_id, request_id, processed, retrieval, timings, snapshot)
        return SearchResponse(
            request_id, trace_id, processed, retrieval, results, timings, snapshot
        )

    def answer(
        self,
        text: str,
        principal: Principal,
        *,
        filters: MetadataFilters | None = None,
        request_id: str | None = None,
        persist_trace: bool = True,
    ) -> AnswerResponse:
        t0 = time.perf_counter()
        search = self.search(
            text,
            principal,
            mode="hybrid",
            filters=filters,
            request_id=request_id,
            persist_trace=persist_trace,
        )
        retrieval, processed = search.retrieval, search.query
        flags = list(retrieval.flags)
        if not retrieval.sufficient:
            return AnswerResponse(
                search.request_id,
                search.trace_id,
                "none",
                None,
                [],
                retrieval.confidence,
                retrieval.reason or "insufficient_evidence",
                [*flags, "insufficient_evidence"],
                None,
                retrieval,
                processed,
                search.timings_ms,
            )
        t_c = time.perf_counter()
        ctx = self.context_builder.build(retrieval.candidates)
        cited: CitedAnswer = extractive_answer(ctx)
        timings = {
            **search.timings_ms,
            "context": round((time.perf_counter() - t_c) * 1000, 1),
            "total": round((time.perf_counter() - t0) * 1000, 1),
        }
        return AnswerResponse(
            search.request_id,
            search.trace_id,
            cited.mode,
            cited.answer,
            [c.as_dict() for c in cited.citations],
            retrieval.confidence,
            None,
            [*flags, *ctx.flags, *cited.flags],
            ctx,
            retrieval,
            processed,
            timings,
        )

    # -- internals -------------------------------------------------------------------------------
    def _persist(
        self,
        trace_id: str,
        request_id: str,
        q: ProcessedQuery,
        r: RetrievalResult,
        timings: dict[str, float],
        snapshot: str | None,
    ) -> None:
        if self.db is None:
            return
        try:
            with self.db.session() as s:
                RetrievalTraceRepository(s).save(
                    trace_id=trace_id,
                    request_id=request_id,
                    principal_id=q.principal.subject,
                    query=q.original if self.settings.core.log_queries else None,
                    intent=q.intent.value,
                    filters={"hard": q.hard_filters.as_dict(), "soft": q.soft_filters.as_dict()},
                    dense_ids=r.dense_ids,
                    sparse_ids=r.sparse_ids,
                    fused_ids=r.fused_ids,
                    reranked=[
                        {
                            "chunk_id": c.chunk_id,
                            **{k: v for k, v in c.scores().items() if v is not None},
                        }
                        for c in r.candidates[:20]
                    ],
                    confidence=r.confidence,
                    gold_snapshot_id=snapshot,
                    timings=timings,
                )
        except Exception:  # trace persistence must never fail a request
            self.flags.append("trace_not_persisted")

    def exclude_deleted(self, chunk_ids: Sequence[str]) -> None:
        if self.bm25 is not None:
            self.bm25.exclude(chunk_ids)


def _locator(p: dict[str, Any]) -> str:
    from akl.rag.context_builder import locator

    return locator(p)
