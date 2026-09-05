"""RAGService (PRD §6.15): wires query processing, hybrid retrieval, context and citations.

Generation (LLM provider, prompt, streaming, memory) arrives with the Chat API milestones;
until then ``answer()`` uses the extractive mode (ADR-010), which is also the permanent
fallback when no LLM is configured.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
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
from akl.rag.citations import CitedAnswer, attach_citations, extractive_answer
from akl.rag.context_builder import BuiltContext, ContextBuilder
from akl.rag.llm.provider import LLMProvider, LLMUnavailableError, build_llm
from akl.rag.memory import ConversationMemory, ConversationNotFoundError, MemoryState
from akl.rag.prompt import PromptBuilder
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
    conversation_id: str | None = None
    rewritten_query: str | None = None
    llm_model: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    def as_dict(self, *, include_trace: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "conversation_id": self.conversation_id,
            "mode": self.mode,
            "answer": self.answer,
            "citations": self.citations,
            "confidence": self.confidence,
            "reason": self.reason,
            "flags": self.flags,
            "retrieval": {
                "dense_k": len(self.retrieval.dense_ids),
                "sparse_k": len(self.retrieval.sparse_ids),
                "fused_k": len(self.retrieval.fused_ids),
                "reranked": self.retrieval.reranker is not None,
                "rewritten_query": self.rewritten_query,
                "intent": self.query.intent.value,
                "filters": {
                    "hard": self.query.hard_filters.as_dict(),
                    "soft": self.query.soft_filters.as_dict(),
                },
            },
            "llm": {"model": self.llm_model, **self.usage} if self.llm_model else None,
            "timings_ms": self.timings_ms,
        }
        if include_trace:
            out["trace"] = self.retrieval.trace()
        return out


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
        llm: LLMProvider | None | str = "auto",
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
        self.sparse_error: str | None = None
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
        self.llm: LLMProvider | None = (
            build_llm(settings.llm)
            if llm == "auto"
            else (llm if not isinstance(llm, str) else None)
        )
        self.prompt = PromptBuilder(
            Path(settings.core.config_dir) / "prompts",
            org_name=settings.core.org_name,
            count_tokens=self.counter.count,
            max_input_tokens=settings.llm.llm_max_input_tokens,
        )
        self.memory: ConversationMemory | None = (
            ConversationMemory(db, settings.llm, self.prompt, self.llm, self.counter.count)
            if db is not None
            else None
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
        conversation_id: uuid.UUID | None = None,
        mode: str = "auto",
        use_memory: bool = True,
    ) -> AnswerResponse:
        """Non-streaming answer: retrieval → (LLM | extractive) → citations → memory (PRD §6.5–6.8)."""
        events = list(
            self._answer_events(
                text,
                principal,
                filters=filters,
                request_id=request_id,
                persist_trace=persist_trace,
                conversation_id=conversation_id,
                mode=mode,
                use_memory=use_memory,
            )
        )
        done = events[-1]
        assert done["event"] == "done"
        response: AnswerResponse = done["response"]
        return response

    def stream_answer(
        self,
        text: str,
        principal: Principal,
        *,
        filters: MetadataFilters | None = None,
        request_id: str | None = None,
        conversation_id: uuid.UUID | None = None,
        mode: str = "auto",
        use_memory: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """Event stream: ``meta`` → ``token``* → ``citations`` → ``done`` (PRD §6.9)."""
        yield from self._answer_events(
            text,
            principal,
            filters=filters,
            request_id=request_id,
            persist_trace=True,
            conversation_id=conversation_id,
            mode=mode,
            use_memory=use_memory,
        )

    def _answer_events(
        self,
        text: str,
        principal: Principal,
        *,
        filters: MetadataFilters | None,
        request_id: str | None,
        persist_trace: bool,
        conversation_id: uuid.UUID | None,
        mode: str,
        use_memory: bool,
    ) -> Iterator[dict[str, Any]]:
        t0 = time.perf_counter()
        request_id = request_id or uuid.uuid4().hex
        state: MemoryState | None = None
        if use_memory and self.memory is not None:
            state = self.memory.load(conversation_id, principal.subject)
        elif conversation_id is not None and self.memory is None:
            raise ConversationNotFoundError(conversation_id)
        retrieval_query = (
            self.memory.rewrite(text, state)
            if (state is not None and self.memory is not None)
            else text
        )

        search = self.search(
            retrieval_query,
            principal,
            mode="hybrid",
            filters=filters,
            request_id=request_id,
            persist_trace=persist_trace,
        )
        retrieval, processed = search.retrieval, search.query
        if state is not None and state.last_citation_chunk_ids:
            for cand in retrieval.candidates:  # PRD §6.10: keep previously cited referents stable
                if cand.chunk_id in state.last_citation_chunk_ids:
                    cand.rrf_score += 0.01
        flags = list(retrieval.flags)
        conv_id = str(state.conversation_id) if state is not None else None
        common = {
            "request_id": search.request_id,
            "trace_id": search.trace_id,
            "conversation_id": conv_id,
        }
        yield {
            "event": "meta",
            **common,
            "intent": processed.intent.value,
            "rewritten_query": retrieval_query if retrieval_query != text else None,
            "confidence": retrieval.confidence,
            "sufficient": retrieval.sufficient,
        }

        def finish(resp: AnswerResponse) -> dict[str, Any]:
            if state is not None and self.memory is not None:
                self.memory.record(
                    state,
                    question=text,
                    rewritten=retrieval_query,
                    answer=resp.answer,
                    mode=resp.mode,
                    confidence=resp.confidence,
                    flags=resp.flags,
                    citations=resp.citations,
                    trace_id=resp.trace_id,
                    request_id=resp.request_id,
                )
            return {
                "event": "done",
                **common,
                "mode": resp.mode,
                "confidence": resp.confidence,
                "reason": resp.reason,
                "flags": resp.flags,
                "timings_ms": resp.timings_ms,
                "response": resp,
            }

        if not retrieval.sufficient:
            resp = AnswerResponse(
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
                conv_id,
                retrieval_query if retrieval_query != text else None,
            )
            yield {"event": "citations", **common, "citations": []}
            yield finish(resp)
            return

        t_c = time.perf_counter()
        ctx = self.context_builder.build(retrieval.candidates)
        timings = {**search.timings_ms, "context": round((time.perf_counter() - t_c) * 1000, 1)}
        use_llm = self.llm is not None and mode in ("auto", "generative")
        cited: CitedAnswer | None = None
        llm_model: str | None = None
        usage: dict[str, Any] = {}

        if use_llm:
            assert self.llm is not None
            prompt = self.prompt.build_answer(
                text,
                ctx,
                summary=state.summary if state else None,
                history=state.history if state else (),
                version=self.settings.llm.prompt_version,
            )
            t_l = time.perf_counter()
            buffer = ""
            full_text = ""
            first_token: float | None = None
            try:
                for delta in self.llm.stream(
                    prompt.messages,
                    max_tokens=self.settings.llm.llm_max_output_tokens,
                    temperature=self.settings.llm.llm_temperature,
                ):
                    if first_token is None:
                        first_token = round((time.perf_counter() - t_l) * 1000, 1)
                    buffer += delta
                    full_text += delta
                    emit, buffer = _split_safe(buffer)
                    if emit:
                        yield {"event": "token", "text": emit}
                if buffer:
                    yield {"event": "token", "text": buffer}
            except LLMUnavailableError as exc:
                flags.append("llm_unavailable")
                self.flags.append(f"llm_unavailable:{exc.message}")
                use_llm = False
            timings["llm_first_token"] = first_token or 0.0
            timings["llm_total"] = round((time.perf_counter() - t_l) * 1000, 1)
            llm_model = self.llm.model
            usage = {
                "input_tokens_est": prompt.input_tokens,
                "output_tokens_est": self.counter.count(full_text),
                "prompt_version": prompt.prompt_version,
            }
            if use_llm:
                cited = self._finalise_generative(full_text, ctx, flags)
                if cited is None:  # model declared insufficient evidence
                    resp = AnswerResponse(
                        search.request_id,
                        search.trace_id,
                        "none",
                        None,
                        [],
                        retrieval.confidence,
                        "model_declared_insufficient",
                        [*flags, "insufficient_evidence"],
                        ctx,
                        retrieval,
                        processed,
                        {**timings, "total": round((time.perf_counter() - t0) * 1000, 1)},
                        conv_id,
                        retrieval_query if retrieval_query != text else None,
                        llm_model,
                        usage,
                    )
                    yield {"event": "citations", **common, "citations": []}
                    yield finish(resp)
                    return
                if (
                    cited.mode == "extractive"
                ):  # downgraded by the faithfulness guards: stream the safe text instead
                    yield {"event": "token", "text": "\n\n" + cited.answer, "replaces": True}

        if cited is None:  # extractive path (no LLM configured or llm failure)
            cited = extractive_answer(ctx)
            yield {"event": "token", "text": cited.answer}

        resp = AnswerResponse(
            search.request_id,
            search.trace_id,
            cited.mode,
            cited.answer,
            [c.as_dict() for c in cited.citations],
            retrieval.confidence,
            None,
            list(dict.fromkeys([*flags, *ctx.flags, *cited.flags])),
            ctx,
            retrieval,
            processed,
            {**timings, "total": round((time.perf_counter() - t0) * 1000, 1)},
            conv_id,
            retrieval_query if retrieval_query != text else None,
            llm_model,
            usage,
        )
        yield {"event": "citations", **common, "citations": resp.citations}
        yield finish(resp)

    def _finalise_generative(
        self, full_text: str, ctx: BuiltContext, flags: list[str]
    ) -> CitedAnswer | None:
        """Apply the hallucination guards of PRD §6.6 to the assembled model output."""
        if "INSUFFICIENT_EVIDENCE" in full_text.upper():
            return None
        cited = attach_citations(
            full_text, ctx, max_uncited_ratio=self.settings.llm.rag_max_uncited_ratio
        )
        cited.mode = "generative"
        unsupported = _unsupported_tokens(full_text, ctx)
        if unsupported:
            cited.flags.append("unsupported_token")
        if (
            "low_faithfulness" in cited.flags and self.settings.llm.rag_strict
        ) or "no_citations" in cited.flags:
            flags.append("downgraded_to_extractive")
            downgraded = extractive_answer(ctx)  # G5: every answer carries citations
            downgraded.flags = list(dict.fromkeys([*downgraded.flags, *cited.flags]))
            return downgraded
        return cited

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

    def reload_indexes(self) -> None:
        """Hot-swap the BM25 index (and its vocabulary) after a rebuild (PRD §7.6 notify_api_reload)."""
        try:
            self.bm25 = Bm25Index.load(
                self.io, k1=self.settings.retrieval.bm25_k1, b=self.settings.retrieval.bm25_b
            )
        except Bm25IndexError as exc:
            self.sparse_error = f"{exc.message} {exc.details}"
            return
        self.sparse_error = None
        self.flags = [f for f in self.flags if f != "sparse_unavailable"]
        self.retriever.sparse = SparseRetriever(self.bm25)
        self.processor = QueryProcessor(
            self.settings.retrieval,
            spell=SpellCorrector.from_texts(p.get("text") or "" for p in self.bm25.payloads),
            known_repos=sorted({str(p["repo"]) for p in self.bm25.payloads if p.get("repo")}),
        )

    def exclude_deleted(self, chunk_ids: Sequence[str]) -> None:
        if self.bm25 is not None:
            self.bm25.exclude(chunk_ids)


def _locator(p: dict[str, Any]) -> str:
    from akl.rag.context_builder import locator

    return locator(p)


_MARKER_TAIL = re.compile(r"\[[\d,\s]*$")
_IDENT = re.compile(
    r"\bAKL_[A-Z0-9_]+\b|\bAKL-[EW]\d{4}\b|\b\d+(?:\.\d+)+\b|\b[\w./-]+\.(?:py|md|yaml|yml|json|sql)\b"
)


def _split_safe(buffer: str) -> tuple[str, str]:
    """Emit everything except a possibly unfinished citation marker at the tail (PRD §6.9)."""
    m = _MARKER_TAIL.search(buffer)
    if m:
        return buffer[: m.start()], buffer[m.start() :]
    return buffer, ""


def _unsupported_tokens(answer: str, ctx: BuiltContext) -> list[str]:
    """Identifiers/paths/versions in the answer that appear in no context block (PRD §6.6)."""
    corpus = (
        "\n".join(b.text for b in ctx.blocks)
        + "\n"
        + "\n".join(str(b.candidate.payload.get("context_prefix") or "") for b in ctx.blocks)
    )
    return sorted({t for t in _IDENT.findall(answer) if t not in corpus})
