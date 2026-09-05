"""Component test: full ingestion flow on the live stack (Milestones 11–15).

Markdown directory → Bronze → Postgres → Silver, with quarantine, near-duplicate
detection, idempotent re-run and deletion tombstones. Everything created is removed.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import delete, text

from akl.config import Settings
from akl.db.models import Document, QuarantineItem
from akl.db.repositories.documents import DocumentRepository
from akl.db.session import Database, DatabaseUnavailableError
from akl.errors import AKLError
from akl.ingestion.connectors.markdown import MarkdownConnector, MarkdownConnectorConfig
from akl.ingestion.service import IngestionService
from akl.lakehouse.engine import DuckDBEngine
from akl.lakehouse.io import Layer

pytestmark = pytest.mark.component


@pytest.fixture
def service() -> Iterator[tuple[IngestionService, list[str]]]:
    try:
        settings = Settings.load()
        db = Database(settings)
        db.ping()
    except (AKLError, DatabaseUnavailableError) as exc:  # pragma: no cover
        pytest.skip(f"stack unavailable: {exc}")
    engine = DuckDBEngine(settings)
    svc = IngestionService(settings, engine, db)
    try:
        svc.io.ensure_bucket()
    except AKLError as exc:  # pragma: no cover
        pytest.skip(f"MinIO unavailable: {exc}")
    run_ids: list[str] = []
    yield svc, run_ids
    # cleanup: Silver/quarantine/manifest files from our runs, Postgres rows, connector state
    for layer, dataset in (
        (Layer.SILVER, "documents"),
        (Layer.SILVER, "chunks"),
        (Layer.SILVER, "dedup_ledger"),
        (Layer.QUARANTINE, "reasons"),
        (Layer.BRONZE, "manifest"),
    ):
        keys = [
            f.key for f in svc.io.list_files(layer, dataset) if any(r in f.key for r in run_ids)
        ]
        if keys:
            svc.io.delete_keys(keys)
    with db.session() as s:
        s.execute(delete(QuarantineItem).where(QuarantineItem.run_id.in_(run_ids)))
        s.execute(delete(Document).where(Document.connector_id.like("ctest-%")))
        s.execute(text("DELETE FROM connector_state WHERE connector_id LIKE 'ctest-%'"))
    engine.close()
    db.dispose()


SENTENCES = [
    "The lakehouse stores raw documents in an immutable bronze layer before any parsing happens.",
    "Silver tables hold cleaned text with character offsets so that citations remain stable over time.",
    "Gold retrieval units are the exact payload contract shared by the vector index and the sparse index.",
    "Every embedding is written to Parquet first, which makes the vector database a rebuildable cache.",
    "Operators can replay any pipeline stage because identifiers are derived deterministically from content.",
    "Quarantined inputs keep their failing rule code so that curators can retry them after a parser fix.",
    "Incremental processing means a run with no upstream changes performs no work at all.",
    "Hybrid retrieval fuses dense and sparse rankings before a cross encoder reorders the candidates.",
]


def _long_text(seed: int, changes: int = 0) -> str:
    """Deterministic English prose (~600 words); ``changes`` edits a few words for a near-duplicate."""
    random.seed(seed)
    body = " ".join(random.choice(SENTENCES) for _ in range(40))
    if changes:
        body = body.replace("lakehouse", "data lakehouse", changes)
    return f"# Document {seed}\n\nIntro paragraph about the platform.\n\n## Details\n\n{body}\n"


def test_full_flow_quarantine_dedup_idempotency_deletion(
    service: tuple[IngestionService, list[str]], tmp_path: Path
) -> None:
    svc, run_ids = service
    tag = uuid.uuid4().hex[:8]
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text(_long_text(1), encoding="utf-8")
    (root / "b-copy.md").write_text(
        _long_text(1, changes=2), encoding="utf-8"
    )  # near-duplicate of a.md
    (root / "short.md").write_text(
        "# Tiny\n\ntoo short\n", encoding="utf-8"
    )  # → AKL-E3005 quarantine
    cfg = MarkdownConnectorConfig(
        id=f"ctest-{tag}",
        type="markdown",
        root_path=root,
        uri_base=f"https://ctest.example.com/{tag}",
        fetch_concurrency=2,
    )

    run1 = f"ctest-{tag}-r1"
    run_ids.append(run1)
    fetch = svc.run_connector(connector=MarkdownConnector(cfg), run_id=run1)
    assert fetch.fetched == 3
    assert fetch.failed == 0
    parse = svc.parse_backlog(run_id=run1)
    assert parse.considered == 3
    assert parse.parsed == 2
    assert parse.quarantined == 1
    assert parse.duplicates == 1
    assert parse.failures[0]["code"] == "AKL-E3005"

    with svc.db.session() as s:
        repo = DocumentRepository(s)
        a = repo.get_by_uri(f"https://ctest.example.com/{tag}/a.md")
        dup = repo.get_by_uri(f"https://ctest.example.com/{tag}/b-copy.md")
        short = repo.get_by_uri(f"https://ctest.example.com/{tag}/short.md")
        assert a is not None
        assert a.status == "silver"
        assert a.current_version_id is not None
        assert dup is not None
        assert dup.status == "silver"
        assert dup.is_duplicate_of == a.document_id
        assert short is not None
        assert short.status == "quarantined"
        versions = repo.versions(a.document_id)
        assert {v.parser_version for v in versions} == {"", "1.0.0"}

    docs = svc.silver.current_documents(
        where=f"canonical_source_uri LIKE 'https://ctest.example.com/{tag}/%'"
    )
    assert docs.num_rows == 2
    by_uri = {r["canonical_source_uri"].rsplit("/", 1)[-1]: r for r in docs.to_pylist()}
    assert by_uri["b-copy.md"]["is_duplicate_of"] == str(a.document_id)
    assert by_uri["a.md"]["language"] == "en"
    assert by_uri["a.md"]["quality_score"] > 0.5
    assert svc.silver.current_documents(where="1=1", columns=["document_id"]).num_rows >= 2

    # idempotent re-run: nothing fetched, nothing parsed
    run2 = f"ctest-{tag}-r2"
    run_ids.append(run2)
    fetch2 = svc.run_connector(connector=MarkdownConnector(cfg), run_id=run2)
    assert fetch2.fetched == 0
    parse2 = svc.parse_backlog(run_id=run2)
    assert parse2.considered == 0

    # deletion at source → tombstone + status deleted
    (root / "b-copy.md").unlink()
    run3 = f"ctest-{tag}-r3"
    run_ids.append(run3)
    fetch3 = svc.run_connector(connector=MarkdownConnector(cfg), run_id=run3)
    assert len(fetch3.deletions) == 1
    with svc.db.session() as s:
        gone = DocumentRepository(s).get_by_uri(f"https://ctest.example.com/{tag}/b-copy.md")
        assert gone is not None
        assert gone.status == "deleted"
    remaining = svc.silver.current_documents(
        where=f"canonical_source_uri LIKE 'https://ctest.example.com/{tag}/%'"
    )
    assert [r["canonical_source_uri"].rsplit("/", 1)[-1] for r in remaining.to_pylist()] == ["a.md"]
