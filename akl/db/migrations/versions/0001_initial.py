"""0001 initial - operational tables."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "pipeline_runs",
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("dag_id", sa.Text(), nullable=False),
        sa.Column("correlation_id", sa.Text(), nullable=True),
        sa.Column("state", sa.Text(), server_default="running", nullable=False),
        sa.Column(
            "started_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("finished_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("conf", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("gold_snapshot_id", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("run_id", name=op.f("pk_pipeline_runs")),
    )
    op.create_index(
        "ix_pipeline_runs_dag_id_started_at",
        "pipeline_runs",
        ["dag_id", sa.text("started_at DESC")],
    )
    op.create_table(
        "task_runs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("task_id", sa.Text(), nullable=False),
        sa.Column("map_index", sa.Integer(), server_default="-1", nullable=False),
        sa.Column("try_number", sa.Integer(), server_default="1", nullable=False),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("started_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("finished_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("rows_in", sa.BigInteger(), nullable=True),
        sa.Column("rows_out", sa.BigInteger(), nullable=True),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["pipeline_runs.run_id"],
            name=op.f("fk_task_runs_run_id_pipeline_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_task_runs")),
    )
    op.create_index("ix_task_runs_run_id", "task_runs", ["run_id"])
    op.create_table(
        "lakehouse_schema_versions",
        sa.Column("dataset", sa.Text(), nullable=False),
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("pyarrow_schema_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "first_written_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint(
            "dataset", "schema_version", name=op.f("pk_lakehouse_schema_versions")
        ),
    )
    op.create_table(
        "lakehouse_files",
        sa.Column("object_key", sa.Text(), nullable=False),
        sa.Column("dataset", sa.Text(), nullable=False),
        sa.Column("partition", sa.Text(), server_default="", nullable=False),
        sa.Column("rows", sa.BigInteger(), nullable=True),
        sa.Column("bytes", sa.BigInteger(), nullable=True),
        sa.Column("run_id", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("object_key", name=op.f("pk_lakehouse_files")),
    )
    op.create_index(
        "ix_lakehouse_files_dataset_partition_is_active",
        "lakehouse_files",
        ["dataset", "partition", "is_active"],
    )
    op.create_table(
        "runtime_config",
        sa.Column("key", sa.Text(), nullable=False),
        sa.Column("value", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("updated_by", sa.Text(), nullable=True),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("key", name=op.f("pk_runtime_config")),
    )


def downgrade() -> None:
    op.drop_table("runtime_config")
    op.drop_index("ix_lakehouse_files_dataset_partition_is_active", table_name="lakehouse_files")
    op.drop_table("lakehouse_files")
    op.drop_table("lakehouse_schema_versions")
    op.drop_index("ix_task_runs_run_id", table_name="task_runs")
    op.drop_table("task_runs")
    op.drop_index("ix_pipeline_runs_dag_id_started_at", table_name="pipeline_runs")
    op.drop_table("pipeline_runs")
