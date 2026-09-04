"""SQLAlchemy ORM models for the AKL metadata catalogue."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, INET, JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
SOURCE_TYPES = ("pdf", "markdown", "html", "github")
DOCUMENT_STATUSES = ("bronze", "silver", "gold", "quarantined", "deleting", "deleted")
SECURITY_LEVELS = ("public", "internal", "restricted")
EMBEDDING_STATUSES = ("pending", "embedded", "failed")
QUARANTINE_STATUSES = ("open", "retried", "dismissed")
SYNC_OPS = ("upsert", "delete")


def _in(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{value}'" for value in values)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {
        datetime: TIMESTAMP(timezone=True),
        dict[str, Any]: JSONB,
        uuid.UUID: UUID(as_uuid=True),
        list[str]: ARRAY(Text),
    }


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    run_id: Mapped[str] = mapped_column(Text, primary_key=True)
    dag_id: Mapped[str] = mapped_column(Text, nullable=False)
    correlation_id: Mapped[str | None] = mapped_column(Text)
    state: Mapped[str] = mapped_column(Text, nullable=False, server_default="running")
    started_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    finished_at: Mapped[datetime | None]
    conf: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    gold_snapshot_id: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    __table_args__ = (Index("ix_pipeline_runs_dag_id_started_at", "dag_id", started_at.desc()),)


class TaskRun(Base):
    __tablename__ = "task_runs"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("pipeline_runs.run_id", ondelete="CASCADE"), nullable=False
    )
    task_id: Mapped[str] = mapped_column(Text, nullable=False)
    map_index: Mapped[int] = mapped_column(Integer, nullable=False, server_default="-1")
    try_number: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    state: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]
    rows_in: Mapped[int | None] = mapped_column(BigInteger)
    rows_out: Mapped[int | None] = mapped_column(BigInteger)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    __table_args__ = (Index("ix_task_runs_run_id", "run_id"),)


class LakehouseSchemaVersion(Base):
    __tablename__ = "lakehouse_schema_versions"
    dataset: Mapped[str] = mapped_column(Text, primary_key=True)
    schema_version: Mapped[str] = mapped_column(Text, primary_key=True)
    pyarrow_schema_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    first_written_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())


class LakehouseFile(Base):
    __tablename__ = "lakehouse_files"
    object_key: Mapped[str] = mapped_column(Text, primary_key=True)
    dataset: Mapped[str] = mapped_column(Text, nullable=False)
    partition: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    rows: Mapped[int | None] = mapped_column(BigInteger)
    bytes: Mapped[int | None] = mapped_column(BigInteger)
    run_id: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    __table_args__ = (
        Index(
            "ix_lakehouse_files_dataset_partition_is_active", "dataset", "partition", "is_active"
        ),
    )


class RuntimeConfig(Base):
    __tablename__ = "runtime_config"
    key: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    updated_by: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now()
    )


class Document(Base):
    __tablename__ = "documents"
    document_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    canonical_source_uri: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    source_type: Mapped[str] = mapped_column(Text, nullable=False)
    connector_id: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="bronze")
    current_version_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(
            "document_versions.document_version_id",
            use_alter=True,
            name="fk_documents_current_version_id_document_versions",
            ondelete="SET NULL",
        )
    )
    latest_content_sha256: Mapped[str | None] = mapped_column(String(64))
    security_level: Mapped[str] = mapped_column(Text, nullable=False, server_default="internal")
    allowed_groups: Mapped[list[str]] = mapped_column(
        nullable=False, server_default=text("'{}'::text[]")
    )
    is_duplicate_of: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("documents.document_id", ondelete="SET NULL")
    )
    pii_types: Mapped[list[str] | None]
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)
    deleted_at: Mapped[datetime | None]
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now()
    )
    __table_args__ = (
        CheckConstraint(f"source_type IN ({_in(SOURCE_TYPES)})", name="source_type"),
        CheckConstraint(f"status IN ({_in(DOCUMENT_STATUSES)})", name="status"),
        CheckConstraint(f"security_level IN ({_in(SECURITY_LEVELS)})", name="security_level"),
        Index("ix_documents_source_type_status", "source_type", "status"),
        Index("ix_documents_connector_id", "connector_id"),
        Index("ix_documents_allowed_groups", "allowed_groups", postgresql_using="gin"),
        Index("ix_documents_updated_at", "updated_at"),
    )


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    document_version_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False
    )
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    text_sha256: Mapped[str | None] = mapped_column(String(64))
    bronze_object_key: Mapped[str] = mapped_column(Text, nullable=False)
    parser_name: Mapped[str | None] = mapped_column(Text)
    parser_version: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    silver_partition: Mapped[str | None] = mapped_column(Text)
    quality_score: Mapped[float | None] = mapped_column(Float)
    quality_flags: Mapped[list[str] | None]
    language: Mapped[str | None] = mapped_column(String(8))
    word_count: Mapped[int | None] = mapped_column(Integer)
    run_id: Mapped[str | None] = mapped_column(Text)
    fetched_at: Mapped[datetime | None]
    parsed_at: Mapped[datetime | None]
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    __table_args__ = (
        UniqueConstraint("document_id", "content_sha256", "parser_version"),
        Index("ix_document_versions_content_sha256", "content_sha256"),
        Index("ix_document_versions_run_id", "run_id"),
    )


class Chunk(Base):
    __tablename__ = "chunks"
    chunk_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    chunk_key: Mapped[str] = mapped_column(String(40), nullable=False)
    lineage_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    chunk_checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    embedded_text_sha256: Mapped[str | None] = mapped_column(String(64))
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False
    )
    document_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_versions.document_version_id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_type: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer)
    quality_score: Mapped[float | None] = mapped_column(Float)
    security_level: Mapped[str] = mapped_column(Text, nullable=False)
    allowed_groups: Mapped[list[str]] = mapped_column(
        nullable=False, server_default=text("'{}'::text[]")
    )
    chunker_version: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_config_hash: Mapped[str] = mapped_column(String(16), nullable=False)
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    embedding_status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now()
    )
    __table_args__ = (
        CheckConstraint(
            f"embedding_status IN ({_in(EMBEDDING_STATUSES)})", name="embedding_status"
        ),
        Index("ix_chunks_document_id_is_current", "document_id", "is_current"),
        Index("ix_chunks_lineage_id", "lineage_id"),
        Index("ix_chunks_chunk_checksum", "chunk_checksum"),
        Index("ix_chunks_embedded_text_sha256", "embedded_text_sha256"),
        Index(
            "ix_chunks_embedding_status_current",
            "embedding_status",
            postgresql_where=text("is_current"),
        ),
        Index(
            "ix_chunks_version_chunker_config",
            "document_version_id",
            "chunker_version",
            "chunk_config_hash",
        ),
    )


class EmbeddingCache(Base):
    __tablename__ = "embedding_cache"
    embedded_text_sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_id: Mapped[str] = mapped_column(Text, primary_key=True)
    model_version: Mapped[str] = mapped_column(Text, primary_key=True)
    dim: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    hit_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    last_hit_at: Mapped[datetime | None]
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    __table_args__ = (Index("ix_embedding_cache_last_hit_at", "last_hit_at"),)


class EmbeddingJob(Base):
    __tablename__ = "embedding_jobs"
    job_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    run_id: Mapped[str | None] = mapped_column(Text)
    embedding_version: Mapped[str] = mapped_column(Text, nullable=False)
    shard: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    chunks_total: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    cache_hits: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    generated: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    failed: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]
    throughput_cps: Mapped[float | None] = mapped_column(Float)
    mlflow_run_id: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    __table_args__ = (Index("ix_embedding_jobs_run_id", "run_id"),)


class EmbeddingBacklog(Base):
    __tablename__ = "embedding_backlog"
    run_id: Mapped[str] = mapped_column(Text, primary_key=True)
    chunk_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    shard: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    __table_args__ = (Index("ix_embedding_backlog_run_id_shard", "run_id", "shard"),)


class QdrantSyncOp(Base):
    __tablename__ = "qdrant_sync_ops"
    run_id: Mapped[str] = mapped_column(Text, primary_key=True)
    op: Mapped[str] = mapped_column(Text, primary_key=True)
    chunk_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    applied: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    __table_args__ = (CheckConstraint(f"op IN ({_in(SYNC_OPS)})", name="op"),)


class LineageEdge(Base):
    __tablename__ = "lineage_edges"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(Text, nullable=False)
    task_id: Mapped[str | None] = mapped_column(Text)
    input_dataset: Mapped[str | None] = mapped_column(Text)
    input_partition: Mapped[str | None] = mapped_column(Text)
    output_dataset: Mapped[str] = mapped_column(Text, nullable=False)
    output_partition: Mapped[str | None] = mapped_column(Text)
    rows_in: Mapped[int | None] = mapped_column(BigInteger)
    rows_out: Mapped[int | None] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    __table_args__ = (
        Index(
            "ix_lineage_edges_output_dataset_output_partition", "output_dataset", "output_partition"
        ),
        Index("ix_lineage_edges_run_id", "run_id"),
    )


class ConnectorState(Base):
    __tablename__ = "connector_state"
    connector_id: Mapped[str] = mapped_column(Text, primary_key=True)
    connector_name: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    last_run_id: Mapped[str | None] = mapped_column(Text)
    last_success_at: Mapped[datetime | None]
    documents_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now()
    )


class QuarantineItem(Base):
    __tablename__ = "quarantine_items"
    quarantine_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    document_id: Mapped[uuid.UUID | None]
    content_sha256: Mapped[str | None] = mapped_column(String(64))
    source_type: Mapped[str | None] = mapped_column(Text)
    error_code: Mapped[str] = mapped_column(Text, nullable=False)
    stage: Mapped[str] = mapped_column(Text, nullable=False)
    detail: Mapped[str | None] = mapped_column(Text)
    object_key: Mapped[str | None] = mapped_column(Text)
    run_id: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="open")
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    resolved_at: Mapped[datetime | None]
    __table_args__ = (
        CheckConstraint(f"status IN ({_in(QUARANTINE_STATUSES)})", name="status"),
        Index("ix_quarantine_items_status_created_at", "status", "created_at"),
        Index("ix_quarantine_items_error_code", "error_code"),
    )


class User(Base):
    __tablename__ = "users"
    user_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    subject: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    email: Mapped[str | None] = mapped_column(Text)
    groups: Mapped[list[str]] = mapped_column(nullable=False, server_default=text("'{}'::text[]"))
    security_levels: Mapped[list[str]] = mapped_column(
        nullable=False, server_default=text("'{public}'::text[]")
    )
    roles: Mapped[list[str]] = mapped_column(
        nullable=False, server_default=text("'{reader}'::text[]")
    )
    disabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    last_login_at: Mapped[datetime | None]


class ApiKey(Base):
    __tablename__ = "api_keys"
    key_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    prefix: Mapped[str] = mapped_column(String(8), nullable=False, unique=True)
    key_hash: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str | None] = mapped_column(Text)
    owner_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.user_id", ondelete="SET NULL")
    )
    scopes: Mapped[list[str]] = mapped_column(nullable=False, server_default=text("'{}'::text[]"))
    groups: Mapped[list[str]] = mapped_column(nullable=False, server_default=text("'{}'::text[]"))
    security_levels: Mapped[list[str]] = mapped_column(
        nullable=False, server_default=text("'{public}'::text[]")
    )
    expires_at: Mapped[datetime | None]
    last_used_at: Mapped[datetime | None]
    revoked_at: Mapped[datetime | None]
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())


class Conversation(Base):
    __tablename__ = "conversations"
    conversation_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    principal_id: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    summary_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    turn_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now()
    )
    expires_at: Mapped[datetime | None]
    __table_args__ = (
        Index("ix_conversations_principal_id_updated_at", "principal_id", "updated_at"),
        Index("ix_conversations_expires_at", "expires_at"),
    )


class Message(Base):
    __tablename__ = "messages"
    message_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"), nullable=False
    )
    turn: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    rewritten_query: Mapped[str | None] = mapped_column(Text)
    token_count: Mapped[int | None] = mapped_column(Integer)
    trace_id: Mapped[str | None] = mapped_column(Text)
    request_id: Mapped[str | None] = mapped_column(Text)
    mode: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    flags: Mapped[list[str] | None]
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    __table_args__ = (Index("ix_messages_conversation_id_turn", "conversation_id", "turn"),)


class AnswerCitation(Base):
    __tablename__ = "answer_citations"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("messages.message_id", ondelete="CASCADE"), nullable=False
    )
    citation_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_id: Mapped[uuid.UUID | None]
    lineage_id: Mapped[uuid.UUID | None]
    document_id: Mapped[uuid.UUID | None]
    locator: Mapped[str | None] = mapped_column(Text)
    score: Mapped[float | None] = mapped_column(Float)
    __table_args__ = (
        Index("ix_answer_citations_message_id", "message_id"),
        Index("ix_answer_citations_chunk_id", "chunk_id"),
    )


class RetrievalTrace(Base):
    __tablename__ = "retrieval_traces"
    trace_id: Mapped[str] = mapped_column(Text, primary_key=True)
    request_id: Mapped[str | None] = mapped_column(Text)
    principal_id: Mapped[str | None] = mapped_column(Text)
    query: Mapped[str | None] = mapped_column(Text)
    intent: Mapped[str | None] = mapped_column(Text)
    filters: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    dense_ids: Mapped[list[str] | None]
    sparse_ids: Mapped[list[str] | None]
    fused_ids: Mapped[list[str] | None]
    reranked: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    confidence: Mapped[float | None] = mapped_column(Float)
    gold_snapshot_id: Mapped[str | None] = mapped_column(Text)
    timings: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    __table_args__ = (Index("ix_retrieval_traces_created_at", "created_at"),)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(primary_key=True, server_default=func.now())
    principal_id: Mapped[str | None] = mapped_column(Text)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    resource_type: Mapped[str | None] = mapped_column(Text)
    resource_id: Mapped[str | None] = mapped_column(Text)
    request_id: Mapped[str | None] = mapped_column(Text)
    ip: Mapped[str | None] = mapped_column(INET)
    user_agent: Mapped[str | None] = mapped_column(Text)
    outcome: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    __table_args__ = (
        Index("ix_audit_log_principal_id_ts", "principal_id", "ts"),
        Index("ix_audit_log_action_ts", "action", "ts"),
        Index("ix_audit_log_resource_id", "resource_id"),
        {"postgresql_partition_by": "RANGE (ts)"},
    )


class IdempotencyKey(Base):
    __tablename__ = "idempotency_keys"
    key: Mapped[str] = mapped_column(Text, primary_key=True)
    principal_id: Mapped[str | None] = mapped_column(Text)
    request_hash: Mapped[str | None] = mapped_column(String(64))
    response: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    __table_args__ = (Index("ix_idempotency_keys_created_at", "created_at"),)


class AdminJob(Base):
    __tablename__ = "admin_jobs"
    job_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="queued")
    progress: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    params: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    started_by: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]
    error_code: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())


class PiiMention(Base):
    __tablename__ = "pii_mentions"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[uuid.UUID | None]
    pii_type: Mapped[str] = mapped_column(Text, nullable=False)
    value_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    __table_args__ = (
        Index("ix_pii_mentions_value_hash", "value_hash"),
        Index("ix_pii_mentions_document_id", "document_id"),
    )


class RateLimitBucket(Base):
    __tablename__ = "rate_limit_buckets"
    principal_id: Mapped[str] = mapped_column(Text, primary_key=True)
    route_class: Mapped[str] = mapped_column(Text, primary_key=True)
    tokens: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now()
    )
