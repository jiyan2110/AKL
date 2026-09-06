"""Component test: the eval harness (Milestone 53) against a real, isolated corpus.

Reuses the Batch-C/D pattern (hash provider, private Qdrant collection, private BM25 version):
ingest two documents, chunk/embed/sync, generate a template-based QA set scoped to exactly those
chunks, then run the real eval runner through the real RAGService and check recall/refusal.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import delete

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
from akl.eval.generate_qa import generate_qa_pairs
from akl.eval.runner import run_eval
from akl.ingestion.connectors.markdown import MarkdownConnector, MarkdownConnectorConfig
from akl.ingestion.service import IngestionService
from akl.lakehouse.engine import DuckDBEngine
from akl.lakehouse.io import Layer
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
    run_ids: list[str] = []
    shas: list[str] = []
    qa_version = f"ctest-eval-{tag}"

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
    units = pipeline.gold.active_units(
        where=f"document_id IN ('{doc_ids[0]}', '{doc_ids[1]}')"
    ).to_pylist()
    Bm25Index.build(units, version=r1).save(ingest.io)

    yield {
        "settings": settings,
        "engine": engine,
        "db": db,
        "ingest": ingest,
        "pipeline": pipeline,
        "provider": provider,
        "collection": collection,
        "units": units,
        "qa_version": qa_version,
        "doc_ids": doc_ids,
    }

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
        for rid in run_ids:
            for name in ("index.jsonl.gz", "meta.json"):
                key = f"gold/indexes/bm25/version={rid}/{name}"
                if ingest.io.object_exists(key):
                    ingest.io.delete_keys([key])
        qa_keys = ingest.io.list_keys(f"gold/eval/qa_pairs/version={qa_version}/")
        if qa_keys:
            ingest.io.delete_keys(qa_keys)
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
        engine.close()
        db.dispose()


def test_generate_and_run_eval_against_real_retrieval(stack: dict[str, object]) -> None:
    settings: Settings = stack["settings"]  # type: ignore[assignment]
    engine: DuckDBEngine = stack["engine"]  # type: ignore[assignment]
    db: Database = stack["db"]  # type: ignore[assignment]
    pipeline: EmbeddingPipeline = stack["pipeline"]  # type: ignore[assignment]
    provider: HashEmbeddingProvider = stack["provider"]  # type: ignore[assignment]
    collection: str = stack["collection"]  # type: ignore[assignment]
    units: list[dict[str, Any]] = stack["units"]  # type: ignore[assignment]
    qa_version: str = stack["qa_version"]  # type: ignore[assignment]

    pairs = generate_qa_pairs(
        units, version=qa_version, n=len(units), distractor_ratio=0.25, seed=1
    )
    pipeline.gold.write_qa_pairs([p.as_row() for p in pairs], run_id=qa_version)
    assert pipeline.gold.latest_qa_version() is not None

    loaded = pipeline.gold.read_qa_pairs(version=qa_version).to_pylist()
    assert len(loaded) == len(pairs)
    n_answerable = sum(1 for p in loaded if p["expected_chunk_ids"])
    n_distractor = len(loaded) - n_answerable
    assert n_answerable >= 1
    assert n_distractor >= 1

    principal = Principal(
        f"ctest-eval-{qa_version}",
        frozenset({"eng"}),
        frozenset({"public", "internal", "restricted"}),
    )
    rag = RAGService(
        settings,
        engine,
        db,
        provider=provider,
        reranker=LexicalReranker(),
        bm25=Bm25Index.load(pipeline.io),
        qdrant_collection=collection,
        manage_alias=False,
        qdrant_use_alias=False,
        allow_download=False,
    )

    report = run_eval(
        rag,
        loaded,
        principal=principal,
        k=5,
        thresholds={"recall_at_5": 0.5, "refusal_precision": 0.3},
    )
    assert report.aggregate["n"] == len(loaded)
    # every answerable question is generated FROM one of these exact chunks, so a working hybrid
    # retriever over this tiny two-document corpus should find it comfortably within top 5
    assert report.aggregate["recall_at_5"] >= 0.5
    assert report.passed, report.failures
    assert all("expected_chunk_ids" in row for row in report.per_query)

    strict = run_eval(rag, loaded, principal=principal, k=5, thresholds={"recall_at_5": 1.1})
    assert strict.passed is False
    assert "recall_at_5" in strict.failures[0]
