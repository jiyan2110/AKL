"""``PATCH /v1/admin/documents/{id}/permissions`` — change a document's security level/groups (PRD §9.3)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request

from akl.api.audit import audit
from akl.api.deps import AppState, get_state, scoped
from akl.api.schemas.admin import PermissionsResponse, PermissionsUpdate
from akl.db.repositories.documents import DocumentRepository
from akl.errors import AKLError
from akl.security.principal import Principal

router = APIRouter(prefix="/v1/admin/documents", tags=["admin"])


class DocumentNotFoundError(AKLError):
    code = "AKL-E6030"
    http_status = 404
    retryable = False


@router.patch("/{document_id}/permissions", response_model=PermissionsResponse)
def update_permissions(
    document_id: str,
    body: PermissionsUpdate,
    request: Request,
    state: AppState = Depends(get_state),
    principal: Principal = Depends(scoped("documents:permissions")),
) -> PermissionsResponse:
    """Update ``security_level`` and/or ``allowed_groups``; either field may be omitted to leave it unchanged."""
    if state.db is None:
        raise DocumentNotFoundError("document not found")
    try:
        did = uuid.UUID(document_id)
    except ValueError as exc:
        raise DocumentNotFoundError(
            "invalid document id", details={"document_id": document_id}
        ) from exc
    with state.db.session() as s:
        repo = DocumentRepository(s)
        doc = repo.get(did)
        if doc is None:
            raise DocumentNotFoundError("document not found", details={"document_id": document_id})
        before = {
            "security_level": doc.security_level,
            "allowed_groups": list(doc.allowed_groups or []),
        }
        if body.security_level is not None:
            doc.security_level = body.security_level
        if body.allowed_groups is not None:
            doc.allowed_groups = body.allowed_groups
        after = {
            "security_level": doc.security_level,
            "allowed_groups": list(doc.allowed_groups or []),
        }
    audit(
        state,
        request,
        principal=principal,
        action="document.permissions.update",
        resource_type="document",
        resource_id=document_id,
        details={"before": before, "after": after},
    )
    return PermissionsResponse(
        document_id=document_id,
        security_level=after["security_level"],
        allowed_groups=after["allowed_groups"],
    )
