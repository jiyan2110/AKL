"""Component test: BM25 build + hybrid retrieval + extractive answer on the live stack (Milestones 26–30).

Reuses the Batch-C pattern: hash provider, private Qdrant collection, own documents, full cleanup.
The BM25 artefact is built into a private version id and LATEST is restored afterwards.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import delete, text

from akl.chunking.incremental import ChunkingService
from akl.config import Settings
from akl.db.models import Document, EmbeddingCache, EmbeddingJob, QuarantineItem, RetrievalTrace
from akl.db.repositories.chunks import ChunkRepository
from akl.db.repositories.documents import DocumentRepository
from akl.db.session import Database, DatabaseUnavailableError
from akl.embedding.bm25.index import LATEST_KEY, Bm25Index
from akl.embedding.pipeline import EmbeddingPipeline
from akl.embedding.provider import HashEmbeddingProvider
from akl.embedding.qdrant.reconciler import QdrantReconciler
from akl.embedding.qdrant.schema import make_client
from akl.errors import AKLError
from akl.ingestion.connectors.markdown import MarkdownConnector, MarkdownConnectorConfig
from akl.ingestion.service import IngestionService
from akl.lakehouse.engine import DuckDBEngine
from akl.lakehouse.io import Layer
from akl.rag.query.filters import MetadataFilters
from akl.rag.retrieval.rerank import LexicalReranker
from akl.rag.service import RAGService
from akl.security.principal import Principal

pytestmark = pytest.mark.component

DOC_A = """# Backup Runbook

## Nightly backups

Back up PostgreSQL every night with pg_dump and copy the archive to the backups prefix in MinIO.
Snapshot the Qdrant collection after each successful sync run and keep fourteen days of history.
Restore drills run monthly and must finish inside one hour for one hundred thousand chunks.
"""

DOC_B = """# Onboarding Guide

## First week

New engineers install Docker Desktop, clone the repository and run make up on their first morning.
The example corpus is ingested with make seed and questions are asked through the search endpoint.
Mentors review the pull request template and the conventional commit rules on Friday afternoon.
"""


@pytest.fixture
def stack(tmp_path: Path) -> Iterator[dict[str, object]]:
    try:
        settings = Settings.load()
        db = Database(settings)
        db.ping()
    except (AKLError, DatabaseUnavailableError) as exc:  # pragma: no cover
        pytest.skip(f"stack unavailable: {exc}")
    engine = DuckDBEngine(settings)
    tag = uuid.uuid4().hex[:8]
    ingest = IngestionService(settings, engine, db)
    chunking = ChunkingService(settings, engine, db)
    provider = HashEmbeddingProvider(settings.embedding.embed_dim)
    pipeline = EmbeddingPipeline(settings, engine, db, provider=provider)
    try:
        ingest.io.ensure_bucket()
        client = make_client(settings)
        client.get_collections()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"MinIO/Qdrant unavailable: {exc}")
    collection = f"kb_ctest_{tag}"
    reconciler = QdrantReconciler(
        client, settings, engine, pipeline.gold, collection=collection, manage_alias=False
    )
    root = tmp_path / "docs"
    root.mkdir()
    cfg = MarkdownConnectorConfig(
        id=f"ctest-{tag}",
        type="markdown",
        root_path=root,
        uri_base=f"https://ctest.example.com/{tag}",
        fetch_concurrency=2,
    )
    latest_before = (
        ingest.io.get_object(LATEST_KEY) if ingest.io.object_exists(LATEST_KEY) else None
    )
    ctx: dict[str, object] = {
        "settings": settings,
        "engine": engine,
        "db": db,
        "ingest": ingest,
        "chunking": chunking,
        "pipeline": pipeline,
        "reconciler": reconciler,
        "cfg": cfg,
        "root": root,
        "run_ids": [],
        "shas": [],
        "collection": collection,
        "provider": provider,
    }
    yield ctx
    run_ids: list[str] = ctx["run_ids"]  # type: ignore[assignment]
    shas: list[str] = ctx["shas"]  # type: ignore[assignment]
    try:
        if client.collection_exists(collection):
            client.delete_collection(collection)
    finally:
        for layer, dataset in (
            (Layer.SILVER, "documents"),
            (Layer.SILVER, "chunks"),
            (Layer.GOLD, "retrieval_units"),
            (Layer.GOLD, "chunk_embeddings"),
            (Layer.QUARANTINE, "reasons"),
            (Layer.BRONZE, "manifest"),
        ):
            keys = [
                f.key
                for f in ingest.io.list_files(layer, dataset)
                if any(r in f.key for r in run_ids)
            ]
            if keys:
                ingest.io.delete_keys(keys)
        for rid in run_ids:  # private BM25 artefacts
            for name in ("index.jsonl.gz", "meta.json"):
                key = f"gold/indexes/bm25/version={rid}/{name}"
                if ingest.io.object_exists(key):
                    ingest.io.delete_keys([key])
        if latest_before is not None:
            ingest.io.put_object(LATEST_KEY, latest_before, content_type="text/plain")
        elif ingest.io.object_exists(LATEST_KEY):
            ingest.io.delete_keys([LATEST_KEY])
        with db.session() as s:
            s.execute(delete(QuarantineItem).where(QuarantineItem.run_id.in_(run_ids)))
            s.execute(delete(EmbeddingJob).where(EmbeddingJob.run_id.in_(run_ids)))
            s.execute(delete(RetrievalTrace).where(RetrievalTrace.principal_id == f"ctest-{tag}"))
            if shas:
                s.execute(
                    delete(EmbeddingCache).where(
                        EmbeddingCache.embedded_text_sha256.in_(shas),
                        EmbeddingCache.model_id == provider.model_id,
                    )
                )
            s.execute(delete(Document).where(Document.connector_id.like("ctest-%")))
            s.execute(text("DELETE FROM connector_state WHERE connector_id LIKE 'ctest-%'"))
        engine.close()
        db.dispose()


def test_bm25_hybrid_search_and_extractive_answer(stack: dict[str, object]) -> None:
    settings: Settings = stack["settings"]  # type: ignore[assignment]
    engine: DuckDBEngine = stack["engine"]  # type: ignore[assignment]
    db: Database = stack["db"]  # type: ignore[assignment]
    ingest: IngestionService = stack["ingest"]  # type: ignore[assignment]
    chunking: ChunkingService = stack["chunking"]  # type: ignore[assignment]
    pipeline: EmbeddingPipeline = stack["pipeline"]  # type: ignore[assignment]
    reconciler: QdrantReconciler = stack["reconciler"]  # type: ignore[assignment]
    cfg: MarkdownConnectorConfig = stack["cfg"]  # type: ignore[assignment]
    root: Path = stack["root"]  # type: ignore[assignment]
    run_ids: list[str] = stack["run_ids"]  # type: ignore[assignment]
    shas: list[str] = stack["shas"]  # type: ignore[assignment]
    collection: str = stack["collection"]  # type: ignore[assignment]
    provider: HashEmbeddingProvider = stack["provider"]  # type: ignore[assignment]
    tag = cfg.id.removeprefix("ctest-")

    (root / "backup.md").write_text(DOC_A, encoding="utf-8")
    (root / "onboarding.md").write_text(DOC_B, encoding="utf-8")
    r1 = f"ctest-{cfg.id}-r1"
    run_ids.append(r1)
    ingest.run_connector(connector=MarkdownConnector(cfg), run_id=r1)
    assert ingest.parse_backlog(run_id=r1).parsed == 2
    with db.session() as s:
        repo = DocumentRepository(s)
        doc_ids = [
            str(repo.get_by_uri(f"{cfg.uri_base}/{n}.md").document_id)
            for n in ("backup", "onboarding")
        ]  # type: ignore[union-attr]
    chunking.run(run_id=r1, document_ids=[uuid.UUID(d) for d in doc_ids])
    pipeline.run(run_id=r1, document_ids=doc_ids)
    with db.session() as s:
        for d in doc_ids:
            shas.extend(
                c.embedded_text_sha256
                for c in ChunkRepository(s).current_for_document(uuid.UUID(d))
                if c.embedded_text_sha256
            )
    sync = reconciler.sync(run_id=r1)
    assert sync.drift == 0

    # --- BM25 built from Gold, scoped to our docs for determinism ---------------------------------
    units = pipeline.gold.active_units(
        where=f"document_id IN ('{doc_ids[0]}', '{doc_ids[1]}')"
    ).to_pylist()
    index = Bm25Index.build(units, version=r1)
    index.save(ingest.io)
    assert Bm25Index.latest_version(ingest.io) == r1
    loaded = Bm25Index.load(ingest.io)
    assert loaded.size == len(units)

    principal = Principal(
        f"ctest-{tag}", frozenset({"eng"}), frozenset({"public", "internal", "restricted"})
    )
    svc = RAGService(
        settings,
        engine,
        db,
        provider=provider,
        reranker=LexicalReranker(),
        bm25=loaded,
        qdrant_collection=collection,
        manage_alias=False,
        qdrant_use_alias=False,
        allow_download=False,
    )

    res = svc.search("how are nightly postgres backups taken", principal, k=5)
    assert res.retrieval.sufficient
    assert res.results[0]["document_id"] == doc_ids[0]
    assert "pg_dump" in res.results[0]["text"]
    assert res.retrieval.dense_ids
    assert res.retrieval.sparse_ids
    assert res.retrieval.reranker == "lexical"
    assert res.results[0]["locator"].startswith("backup.md")

    sparse = svc.search("pg_dump archive", principal, mode="sparse", k=3)
    assert sparse.results
    assert sparse.results[0]["document_id"] == doc_ids[0]
    dense = svc.search("engineers first morning docker", principal, mode="dense", k=3)
    assert dense.results
    assert dense.results[0]["document_id"] == doc_ids[1]

    filtered = svc.search(
        "nightly backups", principal, filters=MetadataFilters(document_ids=[doc_ids[1]]), k=5
    )
    assert all(r["document_id"] == doc_ids[1] for r in filtered.results)
    public_only = Principal(f"ctest-{tag}", frozenset(), frozenset({"public"}))
    assert svc.search("nightly backups", public_only, k=5).results == []

    ans = svc.answer("how are nightly postgres backups taken", principal)
    assert ans.mode == "extractive"
    assert ans.answer
    assert "pg_dump" in ans.answer
    assert ans.citations
    assert ans.citations[0]["document_id"] == doc_ids[0]
    assert ans.citations[0]["locator"].startswith("backup.md")
    assert "no_citations" not in ans.flags

    with db.session() as s:
        traces = list(
            s.scalars(
                text("SELECT trace_id FROM retrieval_traces WHERE principal_id = :p").bindparams(
                    p=f"ctest-{tag}"
                )
            )
        )  # type: ignore[arg-type]
        assert len(traces) >= 5
