"""RunRepository — pipeline_runs / task_runs bookkeeping (PRD Appendix A.7)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from akl.db.models import PipelineRun, TaskRun
from akl.db.repositories import Repository


class RunRepository(Repository):
    def start_run(
        self,
        run_id: str,
        dag_id: str,
        *,
        correlation_id: str | None = None,
        conf: dict[str, Any] | None = None,
    ) -> None:
        stmt = pg_insert(PipelineRun).values(
            run_id=run_id,
            dag_id=dag_id,
            correlation_id=correlation_id,
            conf=conf or {},
            state="running",
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[PipelineRun.run_id], set_={"state": "running", "conf": conf or {}}
        )
        self.session.execute(stmt)

    def finish_run(self, run_id: str, *, state: str, gold_snapshot_id: str | None = None) -> None:
        run = self.session.get(PipelineRun, run_id)
        if run is None:
            return
        run.state = state
        run.finished_at = datetime.now(UTC)
        if gold_snapshot_id:
            run.gold_snapshot_id = gold_snapshot_id

    def start_task(
        self, run_id: str, task_id: str, *, map_index: int = -1, try_number: int = 1
    ) -> int:
        task = TaskRun(
            run_id=run_id,
            task_id=task_id,
            map_index=map_index,
            try_number=try_number,
            state="running",
            started_at=datetime.now(UTC),
        )
        self.session.add(task)
        self.session.flush()
        return int(task.id)

    def finish_task(
        self,
        task_pk: int,
        *,
        state: str,
        rows_in: int | None = None,
        rows_out: int | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        task = self.session.get(TaskRun, task_pk)
        if task is None:
            return
        task.state = state
        task.finished_at = datetime.now(UTC)
        task.rows_in = rows_in
        task.rows_out = rows_out
        task.metrics = metrics or {}

    def tasks(self, run_id: str) -> list[TaskRun]:
        return list(
            self.session.scalars(
                select(TaskRun).where(TaskRun.run_id == run_id).order_by(TaskRun.id)
            )
        )

    def last_success(self, dag_id: str) -> datetime | None:
        return self.session.scalar(
            select(func.max(PipelineRun.finished_at)).where(
                PipelineRun.dag_id == dag_id, PipelineRun.state == "success"
            )
        )

    def recent(self, dag_id: str | None = None, *, limit: int = 20) -> list[PipelineRun]:
        stmt = select(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(limit)
        if dag_id:
            stmt = stmt.where(PipelineRun.dag_id == dag_id)
        return list(self.session.scalars(stmt))
