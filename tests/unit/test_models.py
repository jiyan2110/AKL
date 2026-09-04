"""Unit tests for ORM metadata (Milestone 9) — no database."""

from __future__ import annotations

import pytest

from akl.db.models import AuditLog, Base, Document

pytestmark = pytest.mark.unit

EXPECTED_TABLES = {
    "pipeline_runs",
    "task_runs",
    "lakehouse_schema_versions",
    "lakehouse_files",
    "runtime_config",
    "documents",
    "document_versions",
    "chunks",
    "embedding_cache",
    "embedding_jobs",
    "embedding_backlog",
    "qdrant_sync_ops",
    "lineage_edges",
    "connector_state",
    "quarantine_items",
    "users",
    "api_keys",
    "conversations",
    "messages",
    "answer_citations",
    "retrieval_traces",
    "audit_log",
    "idempotency_keys",
    "admin_jobs",
    "pii_mentions",
    "rate_limit_buckets",
}


def test_table_inventory_matches_appendix_a() -> None:
    assert set(Base.metadata.tables) == EXPECTED_TABLES


def test_every_table_has_primary_key() -> None:
    missing = [t.name for t in Base.metadata.sorted_tables if not t.primary_key.columns]
    assert missing == []


def test_documents_current_version_fk_uses_alter() -> None:
    fk = next(
        fk for fk in Document.__table__.foreign_keys if fk.column.table.name == "document_versions"
    )
    assert fk.use_alter is True
    assert fk.constraint is not None
    assert fk.constraint.name == "fk_documents_current_version_id_document_versions"


def test_audit_log_is_partitioned_by_ts() -> None:
    assert AuditLog.__table__.kwargs["postgresql_partition_by"] == "RANGE (ts)"
    assert [c.name for c in AuditLog.__table__.primary_key.columns] == ["id", "ts"]


def test_check_constraints_named_by_convention() -> None:
    names = {
        c.name for c in Document.__table__.constraints if c.__class__.__name__ == "CheckConstraint"
    }
    assert names == {
        "ck_documents_source_type",
        "ck_documents_status",
        "ck_documents_security_level",
    }
