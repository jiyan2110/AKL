"""0002 domain tables - Appendix A domain metadata."""

from __future__ import annotations

from alembic import op
from sqlalchemy import Table, text

from akl.db.models import Base

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

_OPERATIONAL = {
    "pipeline_runs",
    "task_runs",
    "lakehouse_schema_versions",
    "lakehouse_files",
    "runtime_config",
}


def _domain_tables() -> list[Table]:
    return [table for name, table in Base.metadata.tables.items() if name not in _OPERATIONAL]


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind, tables=_domain_tables())
    bind.execute(
        text("CREATE TABLE IF NOT EXISTS audit_log_default PARTITION OF audit_log DEFAULT")
    )


def downgrade() -> None:
    bind = op.get_bind()
    bind.execute(text("DROP TABLE IF EXISTS audit_log_default"))
    Base.metadata.drop_all(bind=bind, tables=list(reversed(_domain_tables())))
