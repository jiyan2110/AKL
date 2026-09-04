"""Bronze layer dataset schemas."""

from __future__ import annotations

import pyarrow as pa

from akl.lakehouse.schemas import DatasetSchema

SOURCE_TYPES: tuple[str, ...] = ("pdf", "markdown", "html", "github")

MANIFEST = DatasetSchema(
    name="bronze/manifest",
    version="1.0.0",
    description="One row per object fetched into Bronze; append-only.",
    partition_by=("ingest_date",),
    sort_by=("document_id", "fetched_at"),
    schema=pa.schema(
        [
            pa.field("manifest_id", pa.string(), nullable=False),
            pa.field("document_id", pa.string(), nullable=False),
            pa.field("content_sha256", pa.string(), nullable=False),
            pa.field("source_type", pa.string(), nullable=False),
            pa.field("source_uri", pa.string(), nullable=False),
            pa.field("canonical_source_uri", pa.string(), nullable=False),
            pa.field("object_key", pa.string(), nullable=False),
            pa.field("size_bytes", pa.int64(), nullable=False),
            pa.field("mime_type", pa.string(), nullable=True),
            pa.field("fetched_at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("connector_name", pa.string(), nullable=False),
            pa.field("connector_version", pa.string(), nullable=False),
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("source_metadata", pa.map_(pa.string(), pa.string()), nullable=True),
            pa.field("ingest_date", pa.date32(), nullable=False),
        ]
    ),
)

GITHUB_SNAPSHOTS = DatasetSchema(
    name="bronze/github_snapshots",
    version="1.0.0",
    description="File listing of a repository at a commit.",
    partition_by=("repo", "commit_sha"),
    sort_by=("path",),
    schema=pa.schema(
        [
            pa.field("repo", pa.string(), nullable=False),
            pa.field("commit_sha", pa.string(), nullable=False),
            pa.field("path", pa.string(), nullable=False),
            pa.field("blob_sha", pa.string(), nullable=False),
            pa.field("size_bytes", pa.int64(), nullable=False),
            pa.field("mode", pa.string(), nullable=True),
            pa.field("snapshot_at", pa.timestamp("us", tz="UTC"), nullable=False),
        ]
    ),
)

ALL_BRONZE_SCHEMAS: tuple[DatasetSchema, ...] = (MANIFEST, GITHUB_SNAPSHOTS)
