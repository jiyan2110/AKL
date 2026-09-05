"""Component test: chunking service on the live stack (Milestones 16–20).

Ingest two documents → chunk → Gold promotion → idempotent re-run → edit one
document → incremental diff (modified/unchanged) → tombstones → cleanup.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import delete, text

from akl.chunking.incremental import ChunkingService
from akl.config import Settings
from akl.db.models import Document, QuarantineItem
from akl.db.repositories.chunks import ChunkRepository
from akl.db.repositories.documents import DocumentRepository
from akl.db.session import Database, DatabaseUnavailableError
from akl.errors import AKLError
from akl.ingestion.connectors.markdown import MarkdownConnector, MarkdownConnectorConfig
from akl.ingestion.service import IngestionService
from akl.lakehouse.engine import DuckDBEngine
from akl.lakehouse.io import Layer

pytestmark = pytest.mark.component

SENTENCES = [
    "The lakehouse stores raw documents in an immutable bronze layer before any parsing happens.",
    "Silver tables hold cleaned text with character offsets so that citations remain stable over time.",
    "Gold retrieval units are the exact payload contract shared by the vector index and the sparse index.",
    "Every embedding is written to Parquet first, which makes the vector database a rebuildable cache.",
    "Operators can replay any pipeline stage because identifiers are derived deterministically from content.",
    "Quarantined inputs keep their failing rule code so that curators can retry them after a parser fix.",
]


def _doc(seed: int, *, sections: int = 3, tweak: str | None = None) -> str:
    import random

    rng = random.Random(seed)
    parts = [f"# Document {seed}\n"]
    for s in range(sections):
        body = " ".join(rng.choice(SENTENCES) for _ in range(25))
        parts.append(f"## Section {s}\n\n{body}\n")
    text = "\n".join(parts)
    return text.replace("lakehouse", tweak, 1) if tweak else text


@pytest.fixture
def stack(
    tmp_path: Path,
) -> Iterator[tuple[IngestionService, ChunkingService, MarkdownConnectorConfig, Path, list[str]]]:
    try:
        settings = Settings.load()
        db = Database(settings)
        db.ping()
    except (AKLError, DatabaseUnavailableError) as exc:  # pragma: no cover
        pytest.skip(f"stack unavailable: {exc}")
    engine = DuckDBEngine(settings)
    ingest = IngestionService(settings, engine, db)
    chunking = ChunkingService(settings, engine, db)
    try:
        ingest.io.ensure_bucket()
    except AKLError as exc:  # pragma: no cover
        pytest.skip(f"MinIO unavailable: {exc}")
    tag = uuid.uuid4().hex[:8]
    root = tmp_path / "docs"
    root.mkdir()
    cfg = MarkdownConnectorConfig(
        id=f"ctest-{tag}",
        type="markdown",
        root_path=root,
        uri_base=f"https://ctest.example.com/{tag}",
        fetch_concurrency=2,
    )
    run_ids: list[str] = []
    yield ingest, chunking, cfg, root, run_ids
    for layer, dataset in (
        (Layer.SILVER, "documents"),
        (Layer.SILVER, "chunks"),
        (Layer.SILVER, "dedup_ledger"),
        (Layer.GOLD, "retrieval_units"),
        (Layer.QUARANTINE, "reasons"),
        (Layer.BRONZE, "manifest"),
    ):
        keys = [
            f.key for f in ingest.io.list_files(layer, dataset) if any(r in f.key for r in run_ids)
        ]
        if keys:
            ingest.io.delete_keys(keys)
    with db.session() as s:
        s.execute(delete(QuarantineItem).where(QuarantineItem.run_id.in_(run_ids)))
        s.execute(
            delete(Document).where(Document.connector_id.like("ctest-%"))
        )  # cascades to versions/chunks
        s.execute(text("DELETE FROM connector_state WHERE connector_id LIKE 'ctest-%'"))
    engine.close()
    db.dispose()


def test_chunking_incremental_and_gold_promotion(
    stack: tuple[IngestionService, ChunkingService, MarkdownConnectorConfig, Path, list[str]],
) -> None:
    ingest, chunking, cfg, root, run_ids = stack
    (root / "one.md").write_text(_doc(1), encoding="utf-8")
    (root / "two.md").write_text(_doc(2), encoding="utf-8")

    r1 = f"ctest-{cfg.id}-r1"
    run_ids.append(r1)
    ingest.run_connector(connector=MarkdownConnector(cfg), run_id=r1)
    assert ingest.parse_backlog(run_id=r1).parsed == 2

    with chunking.db.session() as s:
        doc_ids = [
            DocumentRepository(s).get_by_uri(f"{cfg.uri_base}/{n}.md").document_id
            for n in ("one", "two")
        ]  # type: ignore[union-attr]

    rep = chunking.run(run_id=r1, document_ids=doc_ids)
    assert rep.documents_considered == 2
    assert rep.documents_chunked == 2
    assert rep.documents_failed == 0
    assert rep.chunks_written == rep.added
    assert rep.added >= 2
    assert rep.gold_rows_promoted >= rep.added  # includes any other backlog; at least ours
    with chunking.db.session() as s:
        current = ChunkRepository(s).current_for_document(doc_ids[0])
        assert len(current) >= 1
        assert all(c.embedding_status == "pending" for c in current)
    where = f"document_id IN ('{doc_ids[0]}', '{doc_ids[1]}')"
    silver_chunks = chunking.silver.current_chunks(where=where)
    assert silver_chunks.num_rows == rep.added
    gold = chunking.gold.active_units(where=where)
    assert gold.num_rows == rep.added
    assert set(gold.column("gold_snapshot_id").to_pylist()) == {r1}

    # idempotent re-run: backlog empty, nothing written
    r2 = f"ctest-{cfg.id}-r2"
    run_ids.append(r2)
    rep2 = chunking.run(run_id=r2, document_ids=doc_ids)
    assert rep2.documents_considered == 0
    assert rep2.chunks_written == 0
    assert rep2.gold_rows_promoted == 0

    # edit one document → new version → incremental: unchanged sections keep ids, edited one is modified
    (root / "one.md").write_text(_doc(1, tweak="data lakehouse"), encoding="utf-8")
    import os

    os.utime(root / "one.md", (2_000_000_000, 2_000_000_000))
    r3 = f"ctest-{cfg.id}-r3"
    run_ids.append(r3)
    ingest.run_connector(connector=MarkdownConnector(cfg), run_id=r3)
    assert ingest.parse_backlog(run_id=r3).parsed == 1
    rep3 = chunking.run(run_id=r3, document_ids=doc_ids)
    assert rep3.documents_considered == 1
    assert rep3.modified >= 1
    assert rep3.unchanged >= 1
    assert rep3.removed == 0
    assert rep3.chunks_tombstoned == rep3.modified + rep3.moved
    assert rep3.chunks_written == rep3.modified + rep3.moved + rep3.added + rep3.reparented
    assert (
        rep3.reparented == rep3.unchanged
    )  # unchanged chunks re-parented to the new document version
    after = chunking.silver.current_chunks(document_id=str(doc_ids[0]))
    assert after.num_rows == len(
        chunking.silver.current_chunks(document_id=str(doc_ids[0])).to_pylist()
    )
    lineage_before = {c.lineage_id for c in current}
    with chunking.db.session() as s:
        now_current = ChunkRepository(s).current_for_document(doc_ids[0])
        assert {
            c.lineage_id for c in now_current
        } == lineage_before  # lineage preserved through the edit
        assert all(
            c.document_version_id != current[0].document_version_id for c in now_current
        )  # re-stamped to new version
    gold_after = chunking.gold.active_units(where=where)
    assert gold_after.num_rows == silver_chunks.num_rows  # same count, new ids for modified chunks
    assert chunking.gold.embedding_backlog().num_rows >= gold_after.num_rows
