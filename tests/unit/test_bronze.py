"""Unit tests for Bronze identities, schemas, and manifest construction."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pyarrow as pa
import pytest

from akl import ids
from akl.lakehouse.bronze import (
    BronzeError,
    BronzeStore,
    RawPutResult,
    extension_for,
    raw_key,
    sha256_hex,
)
from akl.lakehouse.schemas import DatasetSchema, SchemaEnforcementError, enforce
from akl.lakehouse.schemas.bronze import GITHUB_SNAPSHOTS, MANIFEST

pytestmark = pytest.mark.unit
SHA = "a" * 64


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HTTPS://Docs.Example.com:443/a/b/?x=1#frag", "https://docs.example.com/a/b"),
        ("http://example.com", "http://example.com/"),
        ("http://example.com:8080/x/../y/", "http://example.com:8080/y"),
        ("github://Owner/Repo/main/docs/README.md", "github://owner/Repo/main/docs/README.md"),
        ("file:///C:/data/Doc.PDF", "file:///C:/data/Doc.PDF"),
    ],
)
def test_canonicalize_uri(raw: str, expected: str) -> None:
    assert ids.canonicalize_uri(raw) == expected


def test_canonicalize_rejects_missing_scheme() -> None:
    with pytest.raises(ValueError, match="scheme"):
        ids.canonicalize_uri("docs/readme.md")


def test_document_id_is_deterministic_and_uri_sensitive() -> None:
    value = ids.document_id("https://docs.example.com/a")
    assert value == ids.document_id("https://docs.example.com/a")
    assert value != ids.document_id("https://docs.example.com/b")
    assert isinstance(value, uuid.UUID)


def test_raw_key_layout() -> None:
    assert raw_key("pdf", SHA, ".PDF") == f"bronze/raw/source_type=pdf/sha256={SHA}.pdf"
    with pytest.raises(BronzeError):
        raw_key("docx", SHA, "docx")
    with pytest.raises(BronzeError):
        raw_key("pdf", "nothex", "pdf")


@pytest.mark.parametrize(
    ("mime", "filename", "expected"),
    [
        ("application/pdf", None, "pdf"),
        ("text/markdown; charset=utf-8", None, "md"),
        ("text/html", "page.HTML", "html"),
        (None, "notes.RST", "rst"),
        (None, "blob", "bin"),
    ],
)
def test_extension_for(mime: str | None, filename: str | None, expected: str) -> None:
    assert extension_for(mime, filename) == expected


def test_sha256_hex() -> None:
    assert sha256_hex(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_enforce_orders_and_fills_nullable() -> None:
    dataset = DatasetSchema(
        name="t",
        version="1.0.0",
        schema=pa.schema(
            [pa.field("a", pa.int64(), nullable=False), pa.field("b", pa.string(), nullable=True)]
        ),
    )
    output = enforce(pa.table({"a": pa.array([1, 2], pa.int32())}), dataset)
    assert output.column_names == ["a", "b"]
    assert output.schema.field("a").type == pa.int64()
    assert output.column("b").null_count == 2


def test_enforce_rejects_unknown_missing_and_lossy() -> None:
    dataset = DatasetSchema(
        name="t", version="1.0.0", schema=pa.schema([pa.field("a", pa.int8(), nullable=False)])
    )
    with pytest.raises(SchemaEnforcementError, match="unknown"):
        enforce(pa.table({"a": [1], "zzz": [1]}), dataset)
    with pytest.raises(SchemaEnforcementError, match="missing"):
        enforce(pa.table({}), dataset)
    with pytest.raises(SchemaEnforcementError, match="cast"):
        enforce(pa.table({"a": pa.array([1000], pa.int64())}), dataset)


def test_manifest_row_conforms_to_schema() -> None:
    put = RawPutResult(raw_key("markdown", SHA, "md"), SHA, 42, deduplicated=False)
    row = BronzeStore.build_manifest_row(
        source_uri="https://Docs.Example.com/guide/",
        source_type="markdown",
        put=put,
        connector_name="test",
        connector_version="1.0.0",
        run_id="run-1",
        mime_type="text/markdown",
        fetched_at=datetime(2026, 9, 4, 12, 0, tzinfo=UTC),
        source_metadata={"etag": "abc"},
    )
    table = enforce(pa.Table.from_pylist([row], schema=MANIFEST.schema), MANIFEST)
    assert table.num_rows == 1
    assert table.column("ingest_date").to_pylist()[0].isoformat() == "2026-09-04"
    assert row["canonical_source_uri"] == "https://docs.example.com/guide"
    assert row["document_id"] == str(ids.document_id("https://docs.example.com/guide"))
    assert row["manifest_id"] == str(ids.manifest_id(uuid.UUID(row["document_id"]), SHA, "run-1"))


def test_bronze_schema_metadata() -> None:
    assert MANIFEST.partition_by == ("ingest_date",)
    assert GITHUB_SNAPSHOTS.partition_by == ("repo", "commit_sha")
    assert {"name", "version", "fields"} <= MANIFEST.to_json().keys()
