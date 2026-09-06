"""Admin endpoints (PRD §10.8): index reload, lineage. Governance/audit endpoints arrive later."""

from fastapi import APIRouter

from akl.api.routers.admin import lineage, reload

router = APIRouter()
router.include_router(reload.router)
router.include_router(lineage.router)

__all__ = ["router"]
