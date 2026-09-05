"""``POST /v1/admin/reload-index`` — hot-swap the BM25 index after a pipeline rebuild (PRD §7.6)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from akl.api.deps import AppState, get_state, scoped
from akl.security.principal import Principal

router = APIRouter(prefix="/v1/admin", tags=["admin"])


@router.post("/reload-index")
def reload_index(
    state: AppState = Depends(get_state), principal: Principal = Depends(scoped("admin:reload"))
) -> dict[str, Any]:
    rag = state.rag
    if rag is None or not hasattr(rag, "reload_indexes"):
        return {"reloaded": False, "reason": "service not ready"}
    with state.lock:
        rag.reload_indexes()
    return {
        "reloaded": rag.bm25 is not None,
        "bm25_version": getattr(rag.bm25, "version", None),
        "documents": getattr(rag.bm25, "size", None),
        "error": getattr(rag, "sparse_error", None),
        "by": principal.subject,
    }
