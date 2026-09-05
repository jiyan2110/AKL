"""RetrievalTraceRepository — persists per-request retrieval traces (PRD Appendix A.12)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from akl.db.models import RetrievalTrace
from akl.db.repositories import Repository


class RetrievalTraceRepository(Repository):
    def save(
        self,
        *,
        trace_id: str,
        request_id: str | None,
        principal_id: str,
        query: str | None,
        intent: str,
        filters: dict[str, Any],
        dense_ids: list[str],
        sparse_ids: list[str],
        fused_ids: list[str],
        reranked: list[dict[str, Any]],
        confidence: float,
        gold_snapshot_id: str | None,
        timings: dict[str, Any],
    ) -> None:
        self.session.add(
            RetrievalTrace(
                trace_id=trace_id,
                request_id=request_id,
                principal_id=principal_id,
                query=query,
                intent=intent,
                filters=filters,
                dense_ids=dense_ids,
                sparse_ids=sparse_ids,
                fused_ids=fused_ids,
                reranked={"items": reranked},
                confidence=confidence,
                gold_snapshot_id=gold_snapshot_id,
                timings=timings,
            )
        )

    def count(self) -> int:
        return int(self.session.scalar(select(func.count()).select_from(RetrievalTrace)) or 0)
