"""Unit tests: BM25, fusion, reranking, hybrid engine, context builder, citations (Milestones 26–30)."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from akl.config import RetrievalSettings
from akl.embedding.bm25.index import Bm25Index, Bm25IndexError
from akl.embedding.bm25.tokenizer import tokenize
from akl.embedding.provider import HashEmbeddingProvider
from akl.rag.citations import attach_citations, extractive_answer
from akl.rag.context_builder import ContextBuilder, locator
from akl.rag.query import QueryProcessor
from akl.rag.query.filters import MetadataFilters
from akl.rag.retrieval.dense import DenseRetriever
from akl.rag.retrieval.engine import HybridRetriever, RetrievalUnavailableError
from akl.rag.retrieval.fusion import rrf_fuse
from akl.rag.retrieval.models import Candidate
from akl.rag.retrieval.rerank import LexicalReranker
from akl.rag.retrieval.sparse import SparseRetriever
from akl.security.principal import Principal

pytestmark = pytest.mark.unit

DOCS = [
    (
        "backup",
        "Backup Runbook",
        "Nightly backups",
        "internal",
        [],
        "Back up PostgreSQL every night with pg_dump and copy the archive to the backups prefix in MinIO.",
    ),
    (
        "backup",
        "Backup Runbook",
        "Qdrant snapshots",
        "internal",
        [],
        "Snapshot the Qdrant collection after each successful sync run and keep fourteen days of history.",
    ),
    (
        "onboard",
        "Onboarding Guide",
        "First week",
        "internal",
        [],
        "New engineers install Docker Desktop, clone the repository and run make up on the first morning.",
    ),
    (
        "hr",
        "Payroll",
        "Salaries",
        "restricted",
        ["hr"],
        "Salary bands and payroll calendars are reviewed by the compensation committee every quarter.",
    ),
    (
        "chunk",
        "Chunking",
        "Config",
        "internal",
        [],
        "The chunk_config_hash changes whenever AKL_CHUNK_TARGET_TOKENS or any other chunking setting changes.",
    ),
]


def rows() -> list[dict[str, Any]]:
    out = []
    for i, (doc, title, heading, level, groups, text) in enumerate(DOCS):
        out.append(
            {
                "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"chunk-{i}")),
                "lineage_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"chunk-{i}")),
                "document_id": str(uuid.uuid5(uuid.NAMESPACE_URL, doc)),
                "document_version_id": "v1",
                "chunk_index": i,
                "source_type": "markdown",
                "canonical_source_uri": f"https://docs.example.com/{doc}.md",
                "source_uri": f"https://docs.example.com/{doc}.md",
                "title": title,
                "heading_path": [title, heading],
                "heading_breadcrumb": f"{title} › {heading}",
                "chunk_type": "prose",
                "code_language": None,
                "text": text,
                "context_prefix": f"{title} › {heading}",
                "token_count": 20,
                "page_start": None,
                "page_end": None,
                "line_start": 3,
                "line_end": 4,
                "security_level": level,
                "allowed_groups": groups,
                "repo": None,
                "branch": None,
                "path": f"{doc}.md",
                "document_updated_at": 1_700_000_000,
                "quality_score": 0.9,
                "quality_flags": [],
                "language": "en",
                "gold_snapshot_id": "snap",
                "embedded_text_sha256": "e" * 64,
            }
        )
    return out


class FakeIO:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        self.objects[key] = data

    def get_object(self, key: str) -> bytes:
        from akl.lakehouse.io import LakehouseIOError

        if key not in self.objects:
            raise LakehouseIOError("missing", details={"key": key})
        return self.objects[key]

    def object_exists(self, key: str) -> bool:
        return key in self.objects


class FakeQdrant:
    """Stands in for QdrantReconciler.search using an in-memory Qdrant client."""

    def __init__(self, provider: HashEmbeddingProvider, data: list[dict[str, Any]]) -> None:
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            "t", vectors_config=qm.VectorParams(size=provider.dim, distance=qm.Distance.COSINE)
        )
        vectors = provider.embed_documents([f"{r['context_prefix']}\n{r['text']}" for r in data])
        self.client.upsert(
            "t",
            points=[
                qm.PointStruct(id=r["chunk_id"], vector=v.tolist(), payload=r)
                for r, v in zip(data, vectors, strict=True)
            ],
        )
        self.fail = False

    def search(
        self,
        vector: np.ndarray,
        *,
        limit: int,
        query_filter: qm.Filter | None = None,
        hnsw_ef: int | None = None,
        use_alias: bool = True,
    ) -> list[dict[str, Any]]:
        if self.fail:
            raise RuntimeError("qdrant down")
        res = self.client.query_points(
            "t", query=vector.tolist(), limit=limit, query_filter=query_filter, with_payload=True
        )
        return [
            {"chunk_id": str(p.id), "score": float(p.score), **(p.payload or {})}
            for p in res.points
        ]


def make_engine(
    *,
    dense: bool = True,
    sparse: bool = True,
    reranker: bool = True,
    settings: RetrievalSettings | None = None,
) -> tuple[HybridRetriever, FakeQdrant, Bm25Index]:
    data = rows()
    provider = HashEmbeddingProvider(64)
    fq = FakeQdrant(provider, data)
    index = Bm25Index.build(data, version="v1")
    s = settings or RetrievalSettings(rag_min_confidence=0.2, rag_min_candidates=1)
    eng = HybridRetriever(
        s,
        dense=DenseRetriever(fq, provider) if dense else None,  # type: ignore[arg-type]
        sparse=SparseRetriever(index) if sparse else None,
        reranker=LexicalReranker() if reranker else None,
    )
    return eng, fq, index


def process(text: str, principal: Principal | None = None, **kw: Any):  # type: ignore[no-untyped-def]
    return QueryProcessor(RetrievalSettings()).process(text, principal or Principal.dev(), **kw)


# --------------------------------------------------------------------------- BM25
def test_tokenizer_identifiers_and_stopwords() -> None:
    toks = tokenize(
        "How does ChunkingService handle chunk_config_hash in akl/chunking/incremental.py?"
    )
    assert "chunk_config_hash" in toks
    assert {"chunk", "config", "hash", "chunkingservice", "chunking", "service"} <= set(toks)
    assert "how" not in toks  # stopword dropped on long queries
    assert "how" in tokenize("how to")  # kept on very short queries
    assert tokenize("pipelines running") == tokenize("pipeline runs") or tokenize("pipelines")[
        0
    ].startswith("pipelin")


def test_bm25_search_filters_exact_terms_and_roundtrip() -> None:
    index = Bm25Index.build(rows(), version="v1")
    top = index.search("nightly postgres backup pg_dump", k=3)
    assert "pg_dump" in top[0].payload["text"]
    exact = index.search(
        "what controls the chunk config hash", k=3, exact_terms=["chunk_config_hash"]
    )
    assert "chunk_config_hash" in exact[0].payload["text"]
    visible = index.search(
        "salary payroll", k=3, allowed=lambda p: p["security_level"] != "restricted"
    )
    assert visible == []
    io = FakeIO()
    prefix = index.save(io)  # type: ignore[arg-type]
    assert prefix.endswith("version=v1")
    loaded = Bm25Index.load(io)  # type: ignore[arg-type]
    assert loaded.version == "v1"
    assert loaded.size == index.size
    assert [h.chunk_id for h in loaded.search("nightly backup", k=2)] == [
        h.chunk_id for h in index.search("nightly backup", k=2)
    ]
    index.exclude([top[0].chunk_id])
    assert all(
        h.chunk_id != top[0].chunk_id for h in index.search("nightly postgres backup pg_dump", k=3)
    )
    with pytest.raises(Bm25IndexError):
        Bm25Index.load(FakeIO())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- fusion / rerank
def test_rrf_fusion_math_and_penalties() -> None:
    a = Candidate("a", {"quality_flags": []}, dense_score=0.9, dense_rank=1)
    b = Candidate("b", {"quality_flags": ["marginal"]}, dense_score=0.8, dense_rank=2)
    b2 = Candidate("b", {"quality_flags": ["marginal"]}, sparse_score=5.0, sparse_rank=1)
    c = Candidate("c", {"quality_flags": []}, sparse_score=4.0, sparse_rank=2)
    fused = rrf_fuse([a, b], [b2, c], k=60, fused_k=10, marginal_penalty=0.9, soft_bonus=1.0)
    by_id = {x.chunk_id: x for x in fused}
    assert by_id["b"].rrf_score == pytest.approx((1 / 62 + 1 / 61) * 0.9)
    assert by_id["a"].rrf_score == pytest.approx(1 / 61)
    assert by_id["c"].rrf_score == pytest.approx(1 / 62)
    assert [x.chunk_id for x in fused] == ["b", "a", "c"]
    assert by_id["b"].dense_score == 0.8 and by_id["b"].sparse_score == 5.0  # noqa: PT018
    assert "marginal" in by_id["b"].flags


def test_lexical_reranker_orders_by_overlap() -> None:
    cands = [
        Candidate("x", {"text": "The cat sat on the mat."}),
        Candidate("y", {"text": "Back up PostgreSQL nightly with pg_dump."}),
    ]
    ranked = LexicalReranker().rerank("how do I back up postgresql nightly", cands, top_n=2)
    assert ranked[0].chunk_id == "y"
    assert ranked[0].rerank_score is not None
    assert ranked[0].rerank_score > (ranked[1].rerank_score or 0)


# --------------------------------------------------------------------------- hybrid engine
def test_hybrid_retrieval_end_to_end_and_security() -> None:
    eng, _fq, _ix = make_engine()
    res = eng.retrieve(process("how do I back up postgres nightly"))
    assert res.sufficient
    assert res.candidates[0].payload["heading_breadcrumb"].endswith("Nightly backups")
    assert res.dense_ids
    assert res.sparse_ids
    assert res.reranker == "lexical"
    assert set(res.timings_ms) >= {"embed_query", "dense", "sparse", "fusion", "rerank"}
    reader = Principal("r", frozenset(), frozenset({"public", "internal"}))
    r2 = eng.retrieve(process("salary bands payroll calendar", reader))
    assert all(c.payload["security_level"] != "restricted" for c in r2.candidates)
    hr = Principal("h", frozenset({"hr"}), frozenset({"public", "internal", "restricted"}))
    r3 = eng.retrieve(process("salary bands payroll calendar", hr))
    assert r3.candidates[0].payload["title"] == "Payroll"


def test_modes_confidence_gate_and_degradation() -> None:
    eng, fq, _ix = make_engine()
    dense_only = eng.retrieve(process("qdrant snapshot after sync"), mode="dense")
    assert dense_only.sparse_ids == [] and dense_only.candidates  # noqa: PT018
    sparse_only = eng.retrieve(process("qdrant snapshot after sync"), mode="sparse")
    assert sparse_only.dense_ids == [] and sparse_only.candidates  # noqa: PT018
    strict, _, _ = make_engine(
        settings=RetrievalSettings(rag_min_confidence=0.99, rag_min_candidates=1)
    )
    gated = strict.retrieve(process("zebra quantum lattice"))
    assert not gated.sufficient
    assert gated.reason in ("low_confidence", "no_candidates")
    fq.fail = True
    degraded = eng.retrieve(process("nightly backup"))
    assert "dense_unavailable" in degraded.flags
    assert degraded.candidates  # sparse still served
    only_dense, fq2, _ = make_engine(sparse=False)
    fq2.fail = True
    with pytest.raises(RetrievalUnavailableError):
        only_dense.retrieve(process("nightly backup"))
    with pytest.raises(RetrievalUnavailableError):
        HybridRetriever(RetrievalSettings(), dense=None, sparse=None, reranker=None).retrieve(
            process("x")
        )


def test_soft_filter_relaxes_when_thin() -> None:
    eng, _fq, _ix = make_engine(
        settings=RetrievalSettings(rag_min_confidence=0.0, rag_min_candidates=1, rag_top_k=3)
    )
    q = process("nightly backup", filters=None)
    q.soft_filters = MetadataFilters(source_types=["pdf"])  # nothing is a pdf → first pass empty
    res = eng.retrieve(q)
    assert "soft_filter_relaxed" in res.flags
    assert res.candidates
    assert all(not c.soft_match for c in res.candidates)
    hard = eng.retrieve(process("nightly backup", filters=MetadataFilters(source_types=["pdf"])))
    assert hard.candidates == []  # hard filters are never relaxed


# --------------------------------------------------------------------------- context / citations
def _cands(n: int = 3) -> list[Candidate]:
    data = rows()[:n]
    out = []
    for i, r in enumerate(data):
        c = Candidate(r["chunk_id"], r, rrf_score=0.1 - i * 0.01)
        c.rerank_score = 0.9 - i * 0.1
        out.append(c)
    return out


def test_context_builder_dedupe_order_budget_and_render() -> None:
    cands = _cands(3)
    dup = Candidate(
        "dup", {**cands[0].payload, "chunk_id": "dup", "document_id": "other"}, rrf_score=0.05
    )
    dup.rerank_score = 0.85
    count = lambda t: len(t.split())  # noqa: E731
    ctx = ContextBuilder(count, budget_tokens=1000, top_k=8).build([*cands, dup])
    assert [b.chunk_id for b in ctx.blocks] == [
        c.chunk_id for c in cands
    ]  # dup removed; same doc grouped by chunk_index
    assert ctx.dropped == ["dup"]
    assert "deduplicated" in ctx.flags
    rendered = ctx.render()
    assert rendered.startswith('[1] source=markdown title="Backup Runbook"')
    assert "chunk_id=" in rendered
    tight = ContextBuilder(count, budget_tokens=25, top_k=8).build(cands)
    assert tight.total_tokens <= 25
    assert "budget_truncated" in tight.flags
    assert len(tight.blocks) >= 1


def test_locators_per_source() -> None:
    md = {
        "source_type": "github",
        "repo": "org/docs",
        "path": "guide.md",
        "line_start": 10,
        "line_end": 12,
        "branch": "main",
    }
    assert locator(md) == "org/docs/guide.md#L10-L12 @ main"
    pdf = {"source_type": "pdf", "title": "Handbook", "page_start": 3, "page_end": 4}
    assert locator(pdf) == "Handbook, p. 3–4"
    html = {
        "source_type": "html",
        "canonical_source_uri": "https://w.com/a",
        "text": "one two three four five six seven eight nine",
    }
    assert locator(html) == "https://w.com/a#:~:text=one two three four five six seven eight"


def test_attach_citations_validates_renumbers_and_flags() -> None:
    ctx = ContextBuilder(lambda t: len(t.split()), budget_tokens=1000, top_k=8).build(_cands(3))
    cited = attach_citations(
        "Backups run nightly [3]. Snapshots are kept [3][1]. Unsupported claim here without marker. Bogus [9].",
        ctx,
    )
    assert cited.answer.startswith("Backups run nightly [1]. Snapshots are kept [1][2].")
    assert "[9]" not in cited.answer
    assert [c.index for c in cited.citations] == [1, 2]
    assert cited.citations[0].chunk_id == ctx.blocks[2].chunk_id  # old [3] → new [1]
    assert cited.citations[0].locator.startswith("onboard.md")
    assert "invalid_marker" in cited.flags
    assert cited.uncited_ratio == pytest.approx(1 / 3, abs=0.01)
    assert "low_faithfulness" in cited.flags
    none = attach_citations("No markers at all in this answer text.", ctx)
    assert "no_citations" in none.flags


def test_extractive_answer_always_cites() -> None:
    ctx = ContextBuilder(lambda t: len(t.split()), budget_tokens=1000, top_k=8).build(_cands(3))
    ans = extractive_answer(ctx, passages=2)
    assert ans.mode == "extractive"
    assert len(ans.citations) == 2
    assert "no_citations" not in ans.flags
    assert ans.answer.count("[") == 2
    assert ans.citations[0].snippet
