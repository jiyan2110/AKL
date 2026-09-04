"""Connector cursor repository."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from akl.db.models import ConnectorState
from akl.db.repositories import Repository


class ConnectorStateRepository(Repository):
    def get(self, connector_id: str) -> dict[str, Any]:
        row = self.session.get(ConnectorState, connector_id)
        return dict(row.state) if row is not None else {}

    def save(
        self, connector_id: str, connector_name: str, state: dict[str, Any], *, run_id: str
    ) -> None:
        self.session.execute(
            pg_insert(ConnectorState)
            .values(
                connector_id=connector_id,
                connector_name=connector_name,
                state=state,
                last_run_id=run_id,
            )
            .on_conflict_do_update(
                index_elements=[ConnectorState.connector_id],
                set_={
                    "state": state,
                    "connector_name": connector_name,
                    "last_run_id": run_id,
                    "updated_at": func.now(),
                },
            )
        )

    def mark_success(self, connector_id: str, *, documents_count: int) -> None:
        self.session.execute(
            update(ConnectorState)
            .where(ConnectorState.connector_id == connector_id)
            .values(
                last_success_at=datetime.now(UTC),
                documents_count=documents_count,
                updated_at=func.now(),
            )
        )

    def row(self, connector_id: str) -> ConnectorState | None:
        return self.session.get(ConnectorState, connector_id)
