"""SQLAlchemy engine and session management for PostgreSQL."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from akl.config import Settings
from akl.errors import AKLError

DRIVER = "postgresql+psycopg"


class DatabaseUnavailableError(AKLError):
    code = "AKL-E3023"
    http_status = 503
    retryable = True


@dataclass(frozen=True)
class PingResult:
    server_version: str
    current_user: str
    database: str
    latency_ms: float


class Database:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine: Engine | None = None
        self._async_engine: AsyncEngine | None = None
        self._session_factory: sessionmaker[Session] | None = None
        self._async_session_factory: async_sessionmaker[AsyncSession] | None = None

    @property
    def url(self) -> str:
        return self._settings.db.dsn(DRIVER, reveal=True)

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            db = self._settings.db
            self._engine = create_engine(
                self.url,
                pool_size=db.pool_min,
                max_overflow=max(db.pool_max - db.pool_min, 0),
                pool_pre_ping=True,
                pool_recycle=1800,
                connect_args={"application_name": self._settings.core.service_name},
                future=True,
            )
        return self._engine

    @contextmanager
    def session(self) -> Iterator[Session]:
        if self._session_factory is None:
            self._session_factory = sessionmaker(self.engine, expire_on_commit=False)
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @property
    def async_engine(self) -> AsyncEngine:
        if self._async_engine is None:
            db = self._settings.db
            self._async_engine = create_async_engine(
                self.url,
                pool_size=db.pool_min,
                max_overflow=max(db.pool_max - db.pool_min, 0),
                pool_pre_ping=True,
                pool_recycle=1800,
                connect_args={"application_name": self._settings.core.service_name},
            )
        return self._async_engine

    @asynccontextmanager
    async def async_session(self) -> AsyncIterator[AsyncSession]:
        if self._async_session_factory is None:
            self._async_session_factory = async_sessionmaker(
                self.async_engine, expire_on_commit=False
            )
        session = self._async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    def ping(self) -> PingResult:
        import time

        start = time.perf_counter()
        try:
            with self.engine.connect() as connection:
                row = connection.execute(
                    text("SELECT version(), current_user, current_database()")
                ).one()
        except SQLAlchemyError as exc:
            raise DatabaseUnavailableError(
                "cannot connect to PostgreSQL",
                details={"url": self._settings.db.dsn(DRIVER), "error": str(exc.__cause__ or exc)},
            ) from exc
        return PingResult(
            server_version=str(row[0]).split(" on ")[0],
            current_user=str(row[1]),
            database=str(row[2]),
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    def dispose(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    async def adispose(self) -> None:
        if self._async_engine is not None:
            await self._async_engine.dispose()
            self._async_engine = None
