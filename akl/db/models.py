"""SQLAlchemy ORM models for operational metadata tables."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, Boolean, ForeignKey, Index, Integer, MetaData, Text, func
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {datetime: TIMESTAMP(timezone=True), dict[str, Any]: JSONB}


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
