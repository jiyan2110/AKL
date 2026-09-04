"""Live MinIO component test for Silver views and tombstones."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from akl.config import Settings
from akl.errors import AKLError
from akl.lakehouse.engine import DuckDBEngine
from akl.lakehouse.io import LakehouseIO, Layer
from akl.lakehouse.silver import SilverStore

pytestmark = pytest.mark.component


@pytest.fixture
def live() -> Iterator[tuple[SilverStore, LakehouseIO, str]]:
    try:
        settings = Settings.load()
    except AKLError as exc:
        pytest.skip(f"settings unavailable: {exc}")
    engine = DuckDBEngine(settings)
    io = LakehouseIO(settings, engine)
    try:
        io.ensure_bucket()
    except AKLError as exc:
        engine.close()
        pytest.skip(f"MinIO unavailable: {exc}")
    run_id = f"ctest-{uuid.uuid4().hex[:8]}"
    yield SilverStore(io, engine), io, run_id
    for dataset in ("documents", "chunks"):
        keys = [file.key for file in io.list_files(Layer.SILVER, dataset) if run_id in file.key]
        if keys:
            io.delete_keys(keys)
    engine.close()


def test_write_view_tombstone_roundtrip(live: tuple[SilverStore, LakehouseIO, str]) -> None:
    store, _io, run_id = live
    doc_id = f"ctest-{uuid.uuid4()}"
    version = f"{doc_id}:v1"
    now = datetime.now(UTC)
    store.write_documents(
        [
            SilverStore.prepare_document_row(
                {
                    "document_version_id": version,
                    "document_id": doc_id,
                    "content_sha256": "e" * 64,
                    "source_type": "markdown",
                    "canonical_source_uri": f"https://example.com/{doc_id}",
                    "title": "Component test",
                    "language": "en",
                    "text": "Silver round trip.",
                    "quality_score": 0.95,
                    "security_level": "internal",
                    "parser_name": "ctest",
                    "parser_version": "1.0.0",
                    "parsed_at": now,
                }
            )
        ],
        run_id=run_id,
    )
    store.write_chunks(
        [
            {
                "chunk_id": f"{doc_id}:c{i}",
                "chunk_key": "k" * 40,
                "lineage_id": f"{doc_id}:c{i}",
                "chunk_checksum": "f" * 64,
                "document_id": doc_id,
                "document_version_id": version,
                "chunk_index": i,
                "chunk_type": "prose",
                "text": "Silver",
                "start_char": 0,
                "end_char": 6,
                "token_count": 1,
                "overlap_prev_tokens": 0,
                "quality_score": 0.9,
                "chunker_version": "1.0.0",
                "chunk_config_hash": "0123456789abcdef",
                "security_level": "internal",
                "source_type": "markdown",
                "created_at": now,
                "is_current": True,
                "is_deleted": False,
                "ingest_date": now.date(),
            }
            for i in range(2)
        ],
        run_id=run_id,
    )
    assert store.current_documents(where=f"document_id = '{doc_id}'").num_rows == 1
    assert store.current_chunks(document_id=doc_id).num_rows == 2
    assert store.tombstone_documents([doc_id], run_id=run_id) == (1, 2)
    assert store.current_documents(where=f"document_id = '{doc_id}'").num_rows == 0
    assert store.current_chunks(document_id=doc_id).num_rows == 0
