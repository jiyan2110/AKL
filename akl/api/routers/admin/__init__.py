"""Admin endpoints (PRD §10.8): index reload, lineage, RBAC/permissions, API keys, audit, GDPR."""

from fastapi import APIRouter

from akl.api.routers.admin import audit, gdpr, keys, lineage, permissions, reload

router = APIRouter()
router.include_router(reload.router)
router.include_router(lineage.router)
router.include_router(permissions.router)
router.include_router(keys.router)
router.include_router(audit.router)
router.include_router(gdpr.router)

__all__ = ["router"]
