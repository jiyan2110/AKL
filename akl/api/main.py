"""FastAPI application factory (PRD §10.11 main.py).

``create_app()`` builds the app around an :class:`AppState`. In production the lifespan hook
constructs the state from settings (warming the embedding model, reranker, BM25 index and
Qdrant client); tests pass a prebuilt state with fakes.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from akl import __version__
from akl.api import metrics
from akl.api.deps import AppState, JobRegistry, get_state
from akl.api.errors import install_error_handlers
from akl.api.middleware.ratelimit import TokenBucketLimiter
from akl.api.middleware.request_id import RequestContextMiddleware
from akl.api.routers import chat, documents, health, search, sources
from akl.config import Settings
from akl.db.session import Database
from akl.errors import AKLError
from akl.lakehouse.engine import DuckDBEngine
from akl.security.auth import Authenticator

log = logging.getLogger("akl.api")


def build_state(settings: Settings, *, warm: bool = True) -> AppState:
    """Construct the production AppState; failures are captured in ``ready_error`` (readiness → 503)."""
    from akl.rag.service import RAGService

    limiter = TokenBucketLimiter(
        {"search": settings.api.rate_limit_rpm, "chat": settings.api.rate_limit_chat_rpm},
        default_rpm=settings.api.rate_limit_rpm,
    )
    db = Database(settings)
    engine: DuckDBEngine | None = None
    rag: Any = None
    ready_error: str | None = None
    try:
        engine = DuckDBEngine(settings)
        rag = RAGService(settings, engine, db)
        if warm:
            rag.provider.warm_up()
    except AKLError as exc:
        ready_error = f"{exc.code}: {exc.message}"
        log.error("startup_failed", extra={"code": exc.code, "detail": exc.message})
    except Exception as exc:  # readiness must report, not crash
        ready_error = f"{type(exc).__name__}: {exc}"
        log.exception("startup_failed")
    return AppState(
        settings=settings,
        engine=engine,
        db=db,
        rag=rag,
        authenticator=Authenticator(settings, db),
        limiter=limiter,
        jobs=JobRegistry(),
        version=__version__,
        ready_error=ready_error,
    )


def create_app(
    settings: Settings | None = None, *, state: AppState | None = None, warm: bool = True
) -> FastAPI:
    resolved = settings or (state.settings if state is not None else Settings.load())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.akl = state if state is not None else build_state(resolved, warm=warm)
        try:
            yield
        finally:
            st: AppState = app.state.akl
            if st.engine is not None and state is None:
                st.engine.close()
            if st.db is not None and state is None:
                st.db.dispose()

    app = FastAPI(
        title="Enterprise AI Knowledge Lakehouse API",
        version=__version__,
        description="Multi-source RAG over a Bronze/Silver/Gold lakehouse: upload, search, chat with citations.",
        docs_url="/docs" if resolved.api.openapi_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if resolved.api.openapi_enabled else None,
        lifespan=lifespan,
    )
    if resolved.api.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved.api.cors_origin_list,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID"],
        )
    app.add_middleware(RequestContextMiddleware)
    install_error_handlers(app)
    for router in (health.router, search.router, chat.router, documents.router, sources.router):
        app.include_router(router)

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics(st: AppState = Depends(get_state)) -> Response:
        return Response(
            content=metrics.render(), media_type="text/plain; version=0.0.4; charset=utf-8"
        )

    return app


def get_app() -> FastAPI:  # pragma: no cover - uvicorn factory target
    return create_app()
