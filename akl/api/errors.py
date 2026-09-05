"""Exception handlers → error envelope (PRD §10.10, Appendix G)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from akl.errors import AKLError

log = logging.getLogger("akl.api.errors")
DOCS = "https://github.com/your-org/ai-knowledge-lakehouse/blob/main/docs/errors.md"


def _envelope(
    code: str, message: str, details: dict[str, Any], request_id: str | None, retryable: bool
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details,
            "request_id": request_id,
            "retryable": retryable,
            "docs_url": f"{DOCS}#{code}",
        }
    }


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AKLError)
    async def _akl(request: Request, exc: AKLError) -> JSONResponse:
        rid = getattr(request.state, "request_id", None)
        headers: dict[str, str] = {}
        if exc.http_status == 429:
            headers["Retry-After"] = str(int(float(exc.details.get("retry_after_s", 1)) + 1))
        if exc.http_status >= 500:
            log.warning(
                "akl_error", extra={"request_id": rid, "code": exc.code, "detail": exc.message}
            )
        return JSONResponse(
            status_code=exc.http_status,
            content=_envelope(exc.code, exc.message, exc.details, rid, exc.retryable),
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        rid = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=422,
            content=_envelope(
                "AKL-E6002", "request validation failed", {"errors": exc.errors()}, rid, False
            ),
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        rid = getattr(request.state, "request_id", None)
        log.exception("unhandled_error", extra={"request_id": rid})
        return JSONResponse(
            status_code=500,
            content=_envelope(
                "AKL-E9999", "internal error", {"type": type(exc).__name__}, rid, False
            ),
        )
