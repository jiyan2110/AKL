"""Gold layer dataset schemas."""

from __future__ import annotations

import pyarrow as pa

from akl.lakehouse.schemas import DatasetSchema

DEFAULT_EMBEDDING_VERSION = "bge-small-en-v1.5__1.5__384"
DEFAULT_EMBEDDING_DIM = 384
_str_list = pa.list_(pa.string())
_ts = pa.timestamp("us", tz="UTC")

RETRIEVAL_UNITS = DatasetSchema(
    name="gold/retrieval_units",
    version="1.0.0",
    partition_by=("source_type", "security_level"),
    sort_by=("document_id", "chunk_index"),
    schema=pa.schema(
        [
            pa.field("chunk_id", pa.string(), nullable=False),
            pa.field("chunk_key", pa.string(), nullable=False),
            pa.field("lineage_id", pa.string(), nullable=False),
            pa.field("chunk_checksum", pa.string(), nullable=False),
            pa.field("embedded_text_sha256", pa.string(), nullable=True),
            pa.field("document_id", pa.string(), nullable=False),
            pa.field("document_version_id", pa.string(), nullable=False),
            pa.field("chunk_index", pa.int32(), nullable=False),
            pa.field("source_type", pa.string(), nullable=False),
            pa.field("canonical_source_uri", pa.string(), nullable=False),
            pa.field("source_uri", pa.string(), nullable=True),
            pa.field("title", pa.string(), nullable=True),
            pa.field("heading_path", _str_list, nullable=True),
            pa.field("heading_breadcrumb", pa.string(), nullable=True),
            pa.field("chunk_type", pa.string(), nullable=False),
            pa.field("code_language", pa.string(), nullable=True),
            pa.field("text", pa.string(), nullable=False),
            pa.field("context_prefix", pa.string(), nullable=True),
            pa.field("token_count", pa.int32(), nullable=False),
            pa.field("page_start", pa.int32(), nullable=True),
            pa.field("page_end", pa.int32(), nullable=True),
            pa.field("line_start", pa.int32(), nullable=True),
            pa.field("line_end", pa.int32(), nullable=True),
            pa.field("security_level", pa.string(), nullable=False),
            pa.field("allowed_groups", _str_list, nullable=True),
            pa.field("repo", pa.string(), nullable=True),
            pa.field("branch", pa.string(), nullable=True),
            pa.field("path", pa.string(), nullable=True),
            pa.field("document_updated_at", _ts, nullable=False),
            pa.field("quality_score", pa.float32(), nullable=False),
            pa.field("quality_flags", _str_list, nullable=True),
            pa.field("language", pa.string(), nullable=True),
            pa.field("gold_snapshot_id", pa.string(), nullable=False),
            pa.field("created_at", _ts, nullable=False),
        ]
    ),
)

CHUNK_EMBEDDINGS = DatasetSchema(
    name="gold/chunk_embeddings",
    version="1.0.0",
    partition_by=("embedding_version", "source_type"),
    sort_by=("chunk_id",),
    schema=pa.schema(
        [
            pa.field("chunk_id", pa.string(), nullable=False),
            pa.field("chunk_checksum", pa.string(), nullable=False),
            pa.field("embedded_text_sha256", pa.string(), nullable=False),
            pa.field("embedding_version", pa.string(), nullable=False),
            pa.field("model_id", pa.string(), nullable=False),
            pa.field("model_version", pa.string(), nullable=False),
            pa.field("dim", pa.int16(), nullable=False),
            pa.field("vector", pa.list_(pa.float32()), nullable=False),
            pa.field("embedded_at", _ts, nullable=False),
            pa.field("embedder_version", pa.string(), nullable=False),
            pa.field("mlflow_run_id", pa.string(), nullable=True),
            pa.field("source_type", pa.string(), nullable=False),
        ]
    ),
)

EVAL_QA_PAIRS = DatasetSchema(
    name="gold/eval/qa_pairs",
    version="1.0.0",
    partition_by=("version",),
    sort_by=("qa_id",),
    schema=pa.schema(
        [
            pa.field("qa_id", pa.string(), nullable=False),
            pa.field("question", pa.string(), nullable=False),
            pa.field("expected_chunk_ids", _str_list, nullable=False),
            pa.field("expected_document_id", pa.string(), nullable=True),
            pa.field("reference_answer", pa.string(), nullable=True),
            pa.field("generation_method", pa.string(), nullable=False),
            pa.field("difficulty", pa.string(), nullable=True),
            pa.field("version", pa.string(), nullable=False),
        ]
    ),
)

STATS = DatasetSchema(
    name="gold/stats",
    version="1.0.0",
    partition_by=("snapshot_date",),
    sort_by=("metric", "dimension"),
    schema=pa.schema(
        [
            pa.field("metric", pa.string(), nullable=False),
            pa.field("dimension", pa.string(), nullable=False),
            pa.field("value", pa.float64(), nullable=False),
            pa.field("gold_snapshot_id", pa.string(), nullable=False),
            pa.field("computed_at", _ts, nullable=False),
            pa.field("snapshot_date", pa.date32(), nullable=False),
        ]
    ),
)

ALL_GOLD_SCHEMAS: tuple[DatasetSchema, ...] = (
    RETRIEVAL_UNITS,
    CHUNK_EMBEDDINGS,
    EVAL_QA_PAIRS,
    STATS,
)
