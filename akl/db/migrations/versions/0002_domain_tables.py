"""0002 domain tables — PRD Appendix A.1–A.6, A.8–A.14.

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

TS = postgresql.TIMESTAMP(timezone=True)
UUID = postgresql.UUID(as_uuid=True)
JSONB = postgresql.JSONB(astext_type=sa.Text())
TEXT_ARRAY = postgresql.ARRAY(sa.Text())
NOW = sa.text("now()")
EMPTY_ARRAY = sa.text("'{}'::text[]")


def _created_at() -> sa.Column:  # type: ignore[type-arg]
    return sa.Column("created_at", TS, server_default=NOW, nullable=False)


def upgrade() -> None:
    # --- documents (FK to document_versions added after that table exists) ----
    op.create_table(
        "documents",
        sa.Column("document_id", UUID, nullable=False),
        sa.Column("canonical_source_uri", sa.Text(), nullable=False),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("connector_id", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), server_default="bronze", nullable=False),
        sa.Column("current_version_id", UUID, nullable=True),
        sa.Column("latest_content_sha256", sa.String(64), nullable=True),
        sa.Column("security_level", sa.Text(), server_default="internal", nullable=False),
        sa.Column("allowed_groups", TEXT_ARRAY, server_default=EMPTY_ARRAY, nullable=False),
        sa.Column("is_duplicate_of", UUID, nullable=True),
        sa.Column("pii_types", TEXT_ARRAY, nullable=True),
        sa.Column("metadata", JSONB, nullable=True),
        sa.Column("deleted_at", TS, nullable=True),
        _created_at(),
        sa.Column("updated_at", TS, server_default=NOW, nullable=False),
        sa.CheckConstraint(
            "source_type IN ('pdf', 'markdown', 'html', 'github')",
            name=op.f("ck_documents_source_type"),
        ),
        sa.CheckConstraint(
            "status IN ('bronze', 'silver', 'gold', 'quarantined', 'deleting', 'deleted')",
            name=op.f("ck_documents_status"),
        ),
        sa.CheckConstraint(
            "security_level IN ('public', 'internal', 'restricted')",
            name=op.f("ck_documents_security_level"),
        ),
        sa.ForeignKeyConstraint(
            ["is_duplicate_of"],
            ["documents.document_id"],
            name=op.f("fk_documents_is_duplicate_of_documents"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("document_id", name=op.f("pk_documents")),
        sa.UniqueConstraint("canonical_source_uri", name=op.f("uq_documents_canonical_source_uri")),
    )
    op.create_index("ix_documents_source_type_status", "documents", ["source_type", "status"])
    op.create_index("ix_documents_connector_id", "documents", ["connector_id"])
    op.create_index(
        "ix_documents_allowed_groups", "documents", ["allowed_groups"], postgresql_using="gin"
    )
    op.create_index("ix_documents_updated_at", "documents", ["updated_at"])

    # --- document_versions -----------------------------------------------------
    op.create_table(
        "document_versions",
        sa.Column("document_version_id", UUID, nullable=False),
        sa.Column("document_id", UUID, nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("text_sha256", sa.String(64), nullable=True),
        sa.Column("bronze_object_key", sa.Text(), nullable=False),
        sa.Column("parser_name", sa.Text(), nullable=True),
        sa.Column("parser_version", sa.Text(), server_default="", nullable=False),
        sa.Column("silver_partition", sa.Text(), nullable=True),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("quality_flags", TEXT_ARRAY, nullable=True),
        sa.Column("language", sa.String(8), nullable=True),
        sa.Column("word_count", sa.Integer(), nullable=True),
        sa.Column("run_id", sa.Text(), nullable=True),
        sa.Column("fetched_at", TS, nullable=True),
        sa.Column("parsed_at", TS, nullable=True),
        _created_at(),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.document_id"],
            name=op.f("fk_document_versions_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("document_version_id", name=op.f("pk_document_versions")),
        sa.UniqueConstraint(
            "document_id",
            "content_sha256",
            "parser_version",
            name=op.f("uq_document_versions_document_id_content_sha256_parser_version"),
        ),
    )
    op.create_index("ix_document_versions_content_sha256", "document_versions", ["content_sha256"])
    op.create_index("ix_document_versions_run_id", "document_versions", ["run_id"])
    op.create_foreign_key(
        "fk_documents_current_version_id_document_versions",
        "documents",
        "document_versions",
        ["current_version_id"],
        ["document_version_id"],
        ondelete="SET NULL",
    )

    # --- chunks ----------------------------------------------------------------
    op.create_table(
        "chunks",
        sa.Column("chunk_id", UUID, nullable=False),
        sa.Column("chunk_key", sa.String(40), nullable=False),
        sa.Column("lineage_id", UUID, nullable=False),
        sa.Column("chunk_checksum", sa.String(64), nullable=False),
        sa.Column("embedded_text_sha256", sa.String(64), nullable=True),
        sa.Column("document_id", UUID, nullable=False),
        sa.Column("document_version_id", UUID, nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_type", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("security_level", sa.Text(), nullable=False),
        sa.Column("allowed_groups", TEXT_ARRAY, server_default=EMPTY_ARRAY, nullable=False),
        sa.Column("chunker_version", sa.Text(), nullable=False),
        sa.Column("chunk_config_hash", sa.String(16), nullable=False),
        sa.Column("is_current", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("is_deleted", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("embedding_status", sa.Text(), server_default="pending", nullable=False),
        _created_at(),
        sa.Column("updated_at", TS, server_default=NOW, nullable=False),
        sa.CheckConstraint(
            "embedding_status IN ('pending', 'embedded', 'failed')",
            name=op.f("ck_chunks_embedding_status"),
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.document_id"],
            name=op.f("fk_chunks_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.document_version_id"],
            name=op.f("fk_chunks_document_version_id_document_versions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("chunk_id", name=op.f("pk_chunks")),
    )
    op.create_index("ix_chunks_document_id_is_current", "chunks", ["document_id", "is_current"])
    op.create_index("ix_chunks_lineage_id", "chunks", ["lineage_id"])
    op.create_index("ix_chunks_chunk_checksum", "chunks", ["chunk_checksum"])
    op.create_index("ix_chunks_embedded_text_sha256", "chunks", ["embedded_text_sha256"])
    op.create_index(
        "ix_chunks_embedding_status_current",
        "chunks",
        ["embedding_status"],
        postgresql_where=sa.text("is_current"),
    )
    op.create_index(
        "ix_chunks_version_chunker_config",
        "chunks",
        ["document_version_id", "chunker_version", "chunk_config_hash"],
    )

    # --- embedding_cache / jobs / backlog / qdrant_sync_ops ------------------------
    op.create_table(
        "embedding_cache",
        sa.Column("embedded_text_sha256", sa.String(64), nullable=False),
        sa.Column("model_id", sa.Text(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("dim", sa.SmallInteger(), nullable=False),
        sa.Column("vector", sa.LargeBinary(), nullable=False),
        sa.Column("hit_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_hit_at", TS, nullable=True),
        _created_at(),
        sa.PrimaryKeyConstraint(
            "embedded_text_sha256", "model_id", "model_version", name=op.f("pk_embedding_cache")
        ),
    )
    op.create_index("ix_embedding_cache_last_hit_at", "embedding_cache", ["last_hit_at"])

    op.create_table(
        "embedding_jobs",
        sa.Column("job_id", UUID, nullable=False),
        sa.Column("run_id", sa.Text(), nullable=True),
        sa.Column("embedding_version", sa.Text(), nullable=False),
        sa.Column("shard", sa.Integer(), server_default="0", nullable=False),
        sa.Column("chunks_total", sa.Integer(), server_default="0", nullable=False),
        sa.Column("cache_hits", sa.Integer(), server_default="0", nullable=False),
        sa.Column("generated", sa.Integer(), server_default="0", nullable=False),
        sa.Column("failed", sa.Integer(), server_default="0", nullable=False),
        sa.Column("started_at", TS, nullable=True),
        sa.Column("finished_at", TS, nullable=True),
        sa.Column("throughput_cps", sa.Float(), nullable=True),
        sa.Column("mlflow_run_id", sa.Text(), nullable=True),
        _created_at(),
        sa.PrimaryKeyConstraint("job_id", name=op.f("pk_embedding_jobs")),
    )
    op.create_index("ix_embedding_jobs_run_id", "embedding_jobs", ["run_id"])

    op.create_table(
        "embedding_backlog",
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("chunk_id", UUID, nullable=False),
        sa.Column("shard", sa.Integer(), server_default="0", nullable=False),
        sa.PrimaryKeyConstraint("run_id", "chunk_id", name=op.f("pk_embedding_backlog")),
    )
    op.create_index("ix_embedding_backlog_run_id_shard", "embedding_backlog", ["run_id", "shard"])

    op.create_table(
        "qdrant_sync_ops",
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("op", sa.Text(), nullable=False),
        sa.Column("chunk_id", UUID, nullable=False),
        sa.Column("applied", sa.Boolean(), server_default="false", nullable=False),
        sa.CheckConstraint("op IN ('upsert', 'delete')", name=op.f("ck_qdrant_sync_ops_op")),
        sa.PrimaryKeyConstraint("run_id", "op", "chunk_id", name=op.f("pk_qdrant_sync_ops")),
    )

    # --- lineage_edges / connector_state / quarantine_items ------------------------
    op.create_table(
        "lineage_edges",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("task_id", sa.Text(), nullable=True),
        sa.Column("input_dataset", sa.Text(), nullable=True),
        sa.Column("input_partition", sa.Text(), nullable=True),
        sa.Column("output_dataset", sa.Text(), nullable=False),
        sa.Column("output_partition", sa.Text(), nullable=True),
        sa.Column("rows_in", sa.BigInteger(), nullable=True),
        sa.Column("rows_out", sa.BigInteger(), nullable=True),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_lineage_edges")),
    )
    op.create_index(
        "ix_lineage_edges_output_dataset_output_partition",
        "lineage_edges",
        ["output_dataset", "output_partition"],
    )
    op.create_index("ix_lineage_edges_run_id", "lineage_edges", ["run_id"])

    op.create_table(
        "connector_state",
        sa.Column("connector_id", sa.Text(), nullable=False),
        sa.Column("connector_name", sa.Text(), nullable=False),
        sa.Column("state", JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("last_run_id", sa.Text(), nullable=True),
        sa.Column("last_success_at", TS, nullable=True),
        sa.Column("documents_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("updated_at", TS, server_default=NOW, nullable=False),
        sa.PrimaryKeyConstraint("connector_id", name=op.f("pk_connector_state")),
    )

    op.create_table(
        "quarantine_items",
        sa.Column("quarantine_id", UUID, nullable=False),
        sa.Column("document_id", UUID, nullable=True),
        sa.Column("content_sha256", sa.String(64), nullable=True),
        sa.Column("source_type", sa.Text(), nullable=True),
        sa.Column("error_code", sa.Text(), nullable=False),
        sa.Column("stage", sa.Text(), nullable=False),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("object_key", sa.Text(), nullable=True),
        sa.Column("run_id", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), server_default="open", nullable=False),
        _created_at(),
        sa.Column("resolved_at", TS, nullable=True),
        sa.CheckConstraint(
            "status IN ('open', 'retried', 'dismissed')", name=op.f("ck_quarantine_items_status")
        ),
        sa.PrimaryKeyConstraint("quarantine_id", name=op.f("pk_quarantine_items")),
    )
    op.create_index(
        "ix_quarantine_items_status_created_at", "quarantine_items", ["status", "created_at"]
    )
    op.create_index("ix_quarantine_items_error_code", "quarantine_items", ["error_code"])

    # --- users / api_keys ------------------------------------------------------------
    op.create_table(
        "users",
        sa.Column("user_id", UUID, nullable=False),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("email", sa.Text(), nullable=True),
        sa.Column("groups", TEXT_ARRAY, server_default=EMPTY_ARRAY, nullable=False),
        sa.Column(
            "security_levels",
            TEXT_ARRAY,
            server_default=sa.text("'{public}'::text[]"),
            nullable=False,
        ),
        sa.Column(
            "roles", TEXT_ARRAY, server_default=sa.text("'{reader}'::text[]"), nullable=False
        ),
        sa.Column("disabled", sa.Boolean(), server_default="false", nullable=False),
        _created_at(),
        sa.Column("last_login_at", TS, nullable=True),
        sa.PrimaryKeyConstraint("user_id", name=op.f("pk_users")),
        sa.UniqueConstraint("subject", name=op.f("uq_users_subject")),
    )

    op.create_table(
        "api_keys",
        sa.Column("key_id", UUID, nullable=False),
        sa.Column("prefix", sa.String(8), nullable=False),
        sa.Column("key_hash", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("owner_user_id", UUID, nullable=True),
        sa.Column("scopes", TEXT_ARRAY, server_default=EMPTY_ARRAY, nullable=False),
        sa.Column("groups", TEXT_ARRAY, server_default=EMPTY_ARRAY, nullable=False),
        sa.Column(
            "security_levels",
            TEXT_ARRAY,
            server_default=sa.text("'{public}'::text[]"),
            nullable=False,
        ),
        sa.Column("expires_at", TS, nullable=True),
        sa.Column("last_used_at", TS, nullable=True),
        sa.Column("revoked_at", TS, nullable=True),
        _created_at(),
        sa.ForeignKeyConstraint(
            ["owner_user_id"],
            ["users.user_id"],
            name=op.f("fk_api_keys_owner_user_id_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("key_id", name=op.f("pk_api_keys")),
        sa.UniqueConstraint("prefix", name=op.f("uq_api_keys_prefix")),
    )

    # --- conversations / messages / answer_citations / retrieval_traces -----------------
    op.create_table(
        "conversations",
        sa.Column("conversation_id", UUID, nullable=False),
        sa.Column("principal_id", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("summary_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column("turn_count", sa.Integer(), server_default="0", nullable=False),
        _created_at(),
        sa.Column("updated_at", TS, server_default=NOW, nullable=False),
        sa.Column("expires_at", TS, nullable=True),
        sa.PrimaryKeyConstraint("conversation_id", name=op.f("pk_conversations")),
    )
    op.create_index(
        "ix_conversations_principal_id_updated_at", "conversations", ["principal_id", "updated_at"]
    )
    op.create_index("ix_conversations_expires_at", "conversations", ["expires_at"])

    op.create_table(
        "messages",
        sa.Column("message_id", UUID, nullable=False),
        sa.Column("conversation_id", UUID, nullable=False),
        sa.Column("turn", sa.Integer(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("rewritten_query", sa.Text(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("trace_id", sa.Text(), nullable=True),
        sa.Column("request_id", sa.Text(), nullable=True),
        sa.Column("mode", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("flags", TEXT_ARRAY, nullable=True),
        _created_at(),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.conversation_id"],
            name=op.f("fk_messages_conversation_id_conversations"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("message_id", name=op.f("pk_messages")),
    )
    op.create_index("ix_messages_conversation_id_turn", "messages", ["conversation_id", "turn"])

    op.create_table(
        "answer_citations",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("message_id", UUID, nullable=False),
        sa.Column("citation_index", sa.Integer(), nullable=False),
        sa.Column("chunk_id", UUID, nullable=True),
        sa.Column("lineage_id", UUID, nullable=True),
        sa.Column("document_id", UUID, nullable=True),
        sa.Column("locator", sa.Text(), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["messages.message_id"],
            name=op.f("fk_answer_citations_message_id_messages"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_answer_citations")),
    )
    op.create_index("ix_answer_citations_message_id", "answer_citations", ["message_id"])
    op.create_index("ix_answer_citations_chunk_id", "answer_citations", ["chunk_id"])

    op.create_table(
        "retrieval_traces",
        sa.Column("trace_id", sa.Text(), nullable=False),
        sa.Column("request_id", sa.Text(), nullable=True),
        sa.Column("principal_id", sa.Text(), nullable=True),
        sa.Column("query", sa.Text(), nullable=True),
        sa.Column("intent", sa.Text(), nullable=True),
        sa.Column("filters", JSONB, nullable=True),
        sa.Column("dense_ids", TEXT_ARRAY, nullable=True),
        sa.Column("sparse_ids", TEXT_ARRAY, nullable=True),
        sa.Column("fused_ids", TEXT_ARRAY, nullable=True),
        sa.Column("reranked", JSONB, nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("gold_snapshot_id", sa.Text(), nullable=True),
        sa.Column("timings", JSONB, nullable=True),
        _created_at(),
        sa.PrimaryKeyConstraint("trace_id", name=op.f("pk_retrieval_traces")),
    )
    op.create_index("ix_retrieval_traces_created_at", "retrieval_traces", ["created_at"])

    # --- audit_log (partitioned) ----------------------------------------------------------
    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("ts", TS, server_default=NOW, nullable=False),
        sa.Column("principal_id", sa.Text(), nullable=True),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("resource_type", sa.Text(), nullable=True),
        sa.Column("resource_id", sa.Text(), nullable=True),
        sa.Column("request_id", sa.Text(), nullable=True),
        sa.Column("ip", postgresql.INET(), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("outcome", sa.Text(), nullable=True),
        sa.Column("details", JSONB, nullable=True),
        sa.PrimaryKeyConstraint("id", "ts", name=op.f("pk_audit_log")),
        postgresql_partition_by="RANGE (ts)",
    )
    op.create_index("ix_audit_log_principal_id_ts", "audit_log", ["principal_id", "ts"])
    op.create_index("ix_audit_log_action_ts", "audit_log", ["action", "ts"])
    op.create_index("ix_audit_log_resource_id", "audit_log", ["resource_id"])
    op.execute("CREATE TABLE audit_log_default PARTITION OF audit_log DEFAULT")

    # --- idempotency_keys / admin_jobs / pii_mentions / rate_limit_buckets ------------------
    op.create_table(
        "idempotency_keys",
        sa.Column("key", sa.Text(), nullable=False),
        sa.Column("principal_id", sa.Text(), nullable=True),
        sa.Column("request_hash", sa.String(64), nullable=True),
        sa.Column("response", JSONB, nullable=True),
        _created_at(),
        sa.PrimaryKeyConstraint("key", name=op.f("pk_idempotency_keys")),
    )
    op.create_index("ix_idempotency_keys_created_at", "idempotency_keys", ["created_at"])

    op.create_table(
        "admin_jobs",
        sa.Column("job_id", UUID, nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), server_default="queued", nullable=False),
        sa.Column("progress", sa.Float(), server_default="0", nullable=False),
        sa.Column("params", JSONB, nullable=True),
        sa.Column("result", JSONB, nullable=True),
        sa.Column("started_by", sa.Text(), nullable=True),
        sa.Column("started_at", TS, nullable=True),
        sa.Column("finished_at", TS, nullable=True),
        sa.Column("error_code", sa.Text(), nullable=True),
        _created_at(),
        sa.PrimaryKeyConstraint("job_id", name=op.f("pk_admin_jobs")),
    )

    op.create_table(
        "pii_mentions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("document_id", UUID, nullable=False),
        sa.Column("chunk_id", UUID, nullable=True),
        sa.Column("pii_type", sa.Text(), nullable=False),
        sa.Column("value_hash", sa.String(64), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.document_id"],
            name=op.f("fk_pii_mentions_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_pii_mentions")),
    )
    op.create_index("ix_pii_mentions_value_hash", "pii_mentions", ["value_hash"])
    op.create_index("ix_pii_mentions_document_id", "pii_mentions", ["document_id"])

    op.create_table(
        "rate_limit_buckets",
        sa.Column("principal_id", sa.Text(), nullable=False),
        sa.Column("route_class", sa.Text(), nullable=False),
        sa.Column("tokens", sa.Float(), nullable=False),
        sa.Column("updated_at", TS, server_default=NOW, nullable=False),
        sa.PrimaryKeyConstraint("principal_id", "route_class", name=op.f("pk_rate_limit_buckets")),
    )


def downgrade() -> None:
    op.drop_table("rate_limit_buckets")
    op.drop_table("pii_mentions")
    op.drop_table("admin_jobs")
    op.drop_table("idempotency_keys")
    op.execute("DROP TABLE IF EXISTS audit_log_default")
    op.drop_table("audit_log")
    op.drop_table("retrieval_traces")
    op.drop_table("answer_citations")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_table("api_keys")
    op.drop_table("users")
    op.drop_table("quarantine_items")
    op.drop_table("connector_state")
    op.drop_table("lineage_edges")
    op.drop_table("qdrant_sync_ops")
    op.drop_table("embedding_backlog")
    op.drop_table("embedding_jobs")
    op.drop_table("embedding_cache")
    op.drop_table("chunks")
    op.drop_constraint(
        "fk_documents_current_version_id_document_versions", "documents", type_="foreignkey"
    )
    op.drop_table("document_versions")
    op.drop_table("documents")
