"""Live PostgreSQL tests for Milestone 9 metadata and repositories."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import delete, text

from akl.config import Settings
from akl.db.models import AuditLog, Base, Document
from akl.db.repositories.connector_state import ConnectorStateRepository
from akl.db.repositories.documents import DocumentRepository
from akl.db.session import Database
from akl.errors import AKLError

pytestmark = pytest.mark.component


@pytest.fixture
def db() -> Iterator[Database]:
    try:
        settings = Settings.load()
        database = Database(settings)
        database.ping()
    except AKLError as exc:
        pytest.skip(f"database unavailable: {exc}")
    yield database
    database.dispose()


def test_orm_matches_migrated_schema(db: Database) -> None:
    with db.engine.connect() as connection:
        context = MigrationContext.configure(
            connection, opts={"compare_type": False, "compare_server_default": False}
        )
        diffs = compare_metadata(context, Base.metadata)
    structural = [
        diff
        for diff in diffs
        if isinstance(diff, tuple)
        and diff[0] in {"add_table", "remove_table", "add_column", "remove_column"}
        and not (
            diff[0] == "remove_table" and getattr(diff[1], "name", "").startswith("audit_log_")
        )
    ]
    assert structural == [], f"schema drift: {structural}"


def test_record_bronze_is_idempotent(db: Database) -> None:
    uri = f"https://example.com/ctest/{uuid.uuid4()}"
    kwargs = {
        "canonical_source_uri": uri,
        "source_type": "markdown",
        "connector_id": "ctest",
        "content_sha256": "a" * 64,
        "bronze_object_key": "bronze/raw/source_type=markdown/sha256=" + "a" * 64 + ".md",
        "run_id": "ctest-run",
    }
    try:
        with db.session() as session:
            first = DocumentRepository(session).record_bronze(**kwargs)
        with db.session() as session:
            second = DocumentRepository(session).record_bronze(**kwargs)
        assert first.document_created
        assert first.version_created
        assert not second.document_created
        assert not second.version_created
        with db.session() as session:
            repository = DocumentRepository(session)
            new_version = repository.record_bronze(**{**kwargs, "content_sha256": "b" * 64})
            assert new_version.version_created
            assert not new_version.document_created
            document = repository.get(first.document_id)
            assert document is not None
            assert document.status == "bronze"
            assert document.latest_content_sha256 == "b" * 64
            assert len(repository.versions(first.document_id)) == 2
        with db.session() as session:
            DocumentRepository(session).mark_parsed(
                document_version_id=new_version.document_version_id,
                parser_name="ctest",
                parser_version="1.0.0",
                text_sha256="c" * 64,
                quality_score=0.9,
                quality_flags=[],
                language="en",
                word_count=10,
                silver_partition="source_type=markdown/ingest_date=2026-09-04",
                title="Parsed title",
            )
        with db.session() as session:
            document = DocumentRepository(session).get(first.document_id)
            assert document is not None
            assert document.status == "silver"
            assert document.current_version_id == new_version.document_version_id
    finally:
        with db.session() as session:
            session.execute(delete(Document).where(Document.canonical_source_uri == uri))


def test_audit_log_default_partition_accepts_rows(db: Database) -> None:
    with db.session() as session:
        session.add(AuditLog(action="ctest", principal_id="ctest", outcome="ok", details={"k": 1}))
    with db.session() as session:
        assert (
            session.execute(
                text("SELECT count(*) FROM audit_log_default WHERE action = 'ctest'")
            ).scalar_one()
            >= 1
        )
        session.execute(text("DELETE FROM audit_log WHERE action = 'ctest'"))


def test_connector_state_roundtrip(db: Database) -> None:
    connector_id = f"ctest-{uuid.uuid4().hex[:8]}"
    try:
        with db.session() as session:
            repository = ConnectorStateRepository(session)
            assert repository.get(connector_id) == {}
            repository.save(connector_id, "markdown", {"cursor": "abc", "n": 1}, run_id="r1")
        with db.session() as session:
            repository = ConnectorStateRepository(session)
            assert repository.get(connector_id) == {"cursor": "abc", "n": 1}
            repository.mark_success(connector_id, documents_count=3)
        with db.session() as session:
            row = ConnectorStateRepository(session).row(connector_id)
            assert row is not None
            assert row.documents_count == 3
            assert row.last_success_at is not None
    finally:
        with db.session() as session:
            session.execute(
                text("DELETE FROM connector_state WHERE connector_id = :connector_id"),
                {"connector_id": connector_id},
            )
