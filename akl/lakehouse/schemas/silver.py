"""Silver layer dataset schemas."""

from __future__ import annotations

import pyarrow as pa

from akl.lakehouse.schemas import DatasetSchema

SECURITY_LEVELS: tuple[str, ...] = ("public", "internal", "restricted")
CHUNK_TYPES: tuple[str, ...] = ("prose", "code", "table", "heading_only", "list", "mixed")
_str_list = pa.list_(pa.string())
_str_map = pa.map_(pa.string(), pa.string())
_ts = pa.timestamp("us", tz="UTC")

DOCUMENTS = DatasetSchema(
    name="silver/documents",
    version="1.1.0",
    partition_by=("source_type", "ingest_date"),
    sort_by=("document_id", "parsed_at"),
    schema=pa.schema(
        [
            pa.field("document_version_id", pa.string(), nullable=False),
            pa.field("document_id", pa.string(), nullable=False),
            pa.field("content_sha256", pa.string(), nullable=False),
            pa.field("source_type", pa.string(), nullable=False),
            pa.field("source_uri", pa.string(), nullable=True),
            pa.field("canonical_source_uri", pa.string(), nullable=False),
            pa.field("title", pa.string(), nullable=True),
            pa.field("language", pa.string(), nullable=True),
            pa.field("text", pa.string(), nullable=False),
            pa.field("text_sha256", pa.string(), nullable=False),
            pa.field("structure", pa.string(), nullable=True),
            pa.field("tables", pa.string(), nullable=True),
            pa.field("code_blocks", pa.string(), nullable=True),
            pa.field("images", pa.string(), nullable=True),
            pa.field("page_map", pa.string(), nullable=True),
            pa.field("blocks", pa.string(), nullable=True),  # JSON block list (chunking input)
            pa.field("word_count", pa.int32(), nullable=False),
            pa.field("char_count", pa.int32(), nullable=False),
            pa.field("quality_score", pa.float32(), nullable=False),
            pa.field("quality_flags", _str_list, nullable=True),
            pa.field("fingerprint_simhash", pa.uint64(), nullable=True),
            pa.field("is_duplicate_of", pa.string(), nullable=True),
            pa.field("security_level", pa.string(), nullable=False),
            pa.field("allowed_groups", _str_list, nullable=True),
            pa.field("metadata", _str_map, nullable=True),
            pa.field("parser_name", pa.string(), nullable=False),
            pa.field("parser_version", pa.string(), nullable=False),
            pa.field("parsed_at", _ts, nullable=False),
            pa.field("is_current", pa.bool_(), nullable=False),
            pa.field("is_deleted", pa.bool_(), nullable=False),
            pa.field("ingest_date", pa.date32(), nullable=False),
        ]
    ),
)

CHUNKS = DatasetSchema(
    name="silver/chunks",
    version="1.0.0",
    partition_by=("source_type", "ingest_date"),
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
            pa.field("chunk_type", pa.string(), nullable=False),
            pa.field("heading_path", _str_list, nullable=True),
            pa.field("heading_level", pa.int16(), nullable=True),
            pa.field("text", pa.string(), nullable=False),
            pa.field("context_prefix", pa.string(), nullable=True),
            pa.field("start_char", pa.int32(), nullable=False),
            pa.field("end_char", pa.int32(), nullable=False),
            pa.field("page_start", pa.int32(), nullable=True),
            pa.field("page_end", pa.int32(), nullable=True),
            pa.field("line_start", pa.int32(), nullable=True),
            pa.field("line_end", pa.int32(), nullable=True),
            pa.field("token_count", pa.int32(), nullable=False),
            pa.field("overlap_prev_tokens", pa.int32(), nullable=False),
            pa.field("language", pa.string(), nullable=True),
            pa.field("code_language", pa.string(), nullable=True),
            pa.field("quality_score", pa.float32(), nullable=False),
            pa.field("quality_flags", _str_list, nullable=True),
            pa.field("prev_chunk_id", pa.string(), nullable=True),
            pa.field("next_chunk_id", pa.string(), nullable=True),
            pa.field("parent_chunk_id", pa.string(), nullable=True),
            pa.field("chunker_version", pa.string(), nullable=False),
            pa.field("chunk_config_hash", pa.string(), nullable=False),
            pa.field("security_level", pa.string(), nullable=False),
            pa.field("allowed_groups", _str_list, nullable=True),
            pa.field("source_type", pa.string(), nullable=False),
            pa.field("created_at", _ts, nullable=False),
            pa.field("is_current", pa.bool_(), nullable=False),
            pa.field("is_deleted", pa.bool_(), nullable=False),
            pa.field("ingest_date", pa.date32(), nullable=False),
        ]
    ),
)

DEDUP_LEDGER = DatasetSchema(
    name="silver/dedup_ledger",
    version="1.0.0",
    schema=pa.schema(
        [
            pa.field("fingerprint_simhash", pa.uint64(), nullable=False),
            pa.field("canonical_document_id", pa.string(), nullable=False),
            pa.field("duplicate_document_id", pa.string(), nullable=False),
            pa.field("hamming_distance", pa.int16(), nullable=False),
            pa.field("decided_at", _ts, nullable=False),
        ]
    ),
    sort_by=("canonical_document_id",),
)

ALL_SILVER_SCHEMAS: tuple[DatasetSchema, ...] = (DOCUMENTS, CHUNKS, DEDUP_LEDGER)
