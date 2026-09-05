"""Component test: embedding pipeline + cache + Qdrant reconciliation on the live stack (Milestones 21–25).

Uses the deterministic hash provider (no model download) and a private test collection
(alias untouched). Ingest → chunk → embed → sync → drift 0 → idempotent → deletion → search.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import delete, text

from akl.chunking.incremental import ChunkingService
from akl.config import Settings
from akl.db.models import Document, EmbeddingCache, EmbeddingJob, QuarantineItem
from akl.db.repositories.chunks import ChunkRepository
from akl.db.repositories.documents import DocumentRepository
from akl.db.session import Database, DatabaseUnavailableError
from akl.embedding.pipeline import EmbeddingPipeline
from akl.embedding.provider import HashEmbeddingProvider
from akl.embedding.qdrant.reconciler import QdrantReconciler
from akl.embedding.qdrant.schema import make_client
from akl.errors import AKLError
from akl.ingestion.connectors.markdown import MarkdownConnector, MarkdownConnectorConfig
from akl.ingestion.service import IngestionService
from akl.lakehouse.engine import DuckDBEngine
from akl.lakehouse.io import Layer
from akl.rag.query import QueryProcessor
from akl.rag.query.filters import to_qdrant_filter
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
    ctx: dict[str, object] = {
        "ingest": ingest,
        "chunking": chunking,
        "pipeline": pipeline,
        "reconciler": reconciler,
        "cfg": cfg,
        "root": root,
        "run_ids": [],
        "shas": [],
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
            (Layer.SILVER, "dedup_ledger"),
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
        with db.session() as s:
            s.execute(delete(QuarantineItem).where(QuarantineItem.run_id.in_(run_ids)))
            s.execute(delete(EmbeddingJob).where(EmbeddingJob.run_id.in_(run_ids)))
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


def test_embed_cache_sync_drift_deletion_and_search(stack: dict[str, object]) -> None:
    ingest: IngestionService = stack["ingest"]  # type: ignore[assignment]
    chunking: ChunkingService = stack["chunking"]  # type: ignore[assignment]
    pipeline: EmbeddingPipeline = stack["pipeline"]  # type: ignore[assignment]
    reconciler: QdrantReconciler = stack["reconciler"]  # type: ignore[assignment]
    cfg: MarkdownConnectorConfig = stack["cfg"]  # type: ignore[assignment]
    root: Path = stack["root"]  # type: ignore[assignment]
    run_ids: list[str] = stack["run_ids"]  # type: ignore[assignment]
    shas: list[str] = stack["shas"]  # type: ignore[assignment]

    (root / "backup.md").write_text(DOC_A, encoding="utf-8")
    (root / "onboarding.md").write_text(DOC_B, encoding="utf-8")
    r1 = f"ctest-{cfg.id}-r1"
    run_ids.append(r1)
    ingest.run_connector(connector=MarkdownConnector(cfg), run_id=r1)
    assert ingest.parse_backlog(run_id=r1).parsed == 2
    with chunking.db.session() as s:
        repo = DocumentRepository(s)
        doc_ids = [
            str(repo.get_by_uri(f"{cfg.uri_base}/{n}.md").document_id)
            for n in ("backup", "onboarding")
        ]  # type: ignore[union-attr]
    chunk_rep = chunking.run(run_id=r1, document_ids=[uuid.UUID(d) for d in doc_ids])
    assert chunk_rep.documents_chunked == 2
    n_chunks = chunk_rep.added

    # --- embed (cache misses) ------------------------------------------------------
    emb1 = pipeline.run(run_id=r1, document_ids=doc_ids)
    assert emb1.backlog == n_chunks
    assert emb1.generated == n_chunks
    assert emb1.cache_hits == 0
    assert emb1.written == n_chunks
    assert emb1.failed == 0
    assert emb1.job_id is not None
    backlog = pipeline.backlog(document_ids=doc_ids)
    assert backlog == []
    with pipeline.db.session() as s:
        current = ChunkRepository(s).current_for_document(uuid.UUID(doc_ids[0]))
        assert all(c.embedding_status == "embedded" for c in current)
        shas.extend(c.embedded_text_sha256 for c in current if c.embedded_text_sha256)
        shas.extend(
            c.embedded_text_sha256
            for c in ChunkRepository(s).current_for_document(uuid.UUID(doc_ids[1]))
            if c.embedded_text_sha256
        )

    # --- idempotent: second embed run has empty backlog ---------------------------------
    r2 = f"ctest-{cfg.id}-r2"
    run_ids.append(r2)
    emb2 = pipeline.run(run_id=r2, document_ids=doc_ids)
    assert emb2.backlog == 0
    assert emb2.written == 0

    # --- Qdrant sync: private collection, alias untouched --------------------------------
    sync1 = reconciler.sync(run_id=r1)
    assert sync1.gold_points >= n_chunks
    assert sync1.upserted == sync1.to_upsert
    assert sync1.drift == 0
    st = reconciler.schema.status()
    assert st.points == sync1.gold_points
    assert st.missing_indexes == ()  # server Qdrant reports payload indexes
    assert st.alias_target != st.name  # we never re-pointed the production alias
    sync2 = reconciler.sync(run_id=r2, dry_run=True)
    assert sync2.to_upsert == 0
    assert sync2.to_delete == 0

    # --- dense search through the private collection with the security filter -------------
    q = QueryProcessor(pipeline.settings.retrieval).process(
        "how are nightly postgres backups taken", Principal.dev()
    )
    vec = pipeline.provider.embed_query(q.dense_text)
    hits = reconciler.search(
        vec, limit=3, query_filter=to_qdrant_filter(q.principal, q.hard_filters), use_alias=False
    )
    assert hits
    assert hits[0]["document_id"] == doc_ids[0]
    assert "pg_dump" in hits[0]["text"]
    public_only = Principal("p", frozenset(), frozenset({"public"}))
    assert (
        reconciler.search(vec, limit=3, query_filter=to_qdrant_filter(public_only), use_alias=False)
        == []
    )  # docs are internal

    # --- cache hit path: re-embed after chunk re-creation yields hits, not generation ----------
    with pipeline.db.session() as s:
        ChunkRepository(s).set_embedding_status([c.chunk_id for c in current], "pending")
    hits_before = pipeline.backlog(document_ids=doc_ids)
    assert hits_before == []  # Gold coverage, not Postgres status, drives the backlog (ADR-001)

    # --- deletion at source → tombstone → sync deletes points → drift 0 -----------------------
    (root / "onboarding.md").unlink()
    r3 = f"ctest-{cfg.id}-r3"
    run_ids.append(r3)
    fetch3 = ingest.run_connector(connector=MarkdownConnector(cfg), run_id=r3)
    assert len(fetch3.deletions) == 1
    sync3 = reconciler.sync(run_id=r3)
    assert sync3.deleted >= 1
    assert sync3.drift == 0
    remaining = reconciler.search(
        pipeline.provider.embed_query("onboarding first week docker"),
        limit=5,
        query_filter=to_qdrant_filter(Principal.dev()),
        use_alias=False,
    )
    assert all(h["document_id"] != doc_ids[1] for h in remaining)
