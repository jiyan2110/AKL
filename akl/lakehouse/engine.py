"""DuckDB query engine and migration seam."""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any
from urllib.parse import urlparse

import duckdb
import pyarrow as pa

from akl.config import Settings
from akl.errors import AKLError


class LakehouseEngineError(AKLError):
    code = "AKL-E2001"
    http_status = 503
    retryable = True


class QueryEngine(ABC):
    @abstractmethod
    def execute(self, sql: str, params: list[Any] | None = None) -> pa.Table:
        """Run SQL and return an Arrow table."""

    @abstractmethod
    def execute_scalar(self, sql: str, params: list[Any] | None = None) -> Any:
        """Run SQL and return the first scalar."""

    @abstractmethod
    def register(self, name: str, table: pa.Table) -> None:
        """Register an Arrow table as a view."""

    @abstractmethod
    def unregister(self, name: str) -> None:
        """Unregister a view."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""


class DuckDBEngine(QueryEngine):
    S3_SECRET_NAME = "akl_s3"  # noqa: S105 - identifier, not a credential

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._s3_ready = False
        try:
            self._con = duckdb.connect(database=":memory:")
            self._con.execute(f"SET memory_limit = '{settings.lakehouse.duckdb_memory_limit}'")
            self._con.execute(f"SET threads = {settings.lakehouse.duckdb_threads}")
            self._con.execute("SET preserve_insertion_order = false")
        except duckdb.Error as exc:
            raise LakehouseEngineError(
                "failed to initialise DuckDB", details={"error": str(exc)}
            ) from exc

    def ensure_s3(self) -> None:
        """Load httpfs and configure the MinIO S3 secret."""
        if self._s3_ready:
            return
        s3 = self._settings.s3
        parsed = urlparse(s3.endpoint)
        host = parsed.netloc or parsed.path
        use_ssl = "true" if parsed.scheme == "https" or s3.use_ssl else "false"
        url_style = "path" if s3.path_style else "vhost"
        try:
            self._con.execute("INSTALL httpfs")
            self._con.execute("LOAD httpfs")
            self._con.execute(
                f"""
                CREATE OR REPLACE SECRET {self.S3_SECRET_NAME} (
                    TYPE S3, KEY_ID ?, SECRET ?, ENDPOINT ?, REGION ?,
                    URL_STYLE '{url_style}', USE_SSL {use_ssl}
                )
                """,
                [
                    s3.access_key.get_secret_value(),
                    s3.secret_key.get_secret_value(),
                    host,
                    s3.region,
                ],
            )
        except duckdb.Error as exc:
            raise LakehouseEngineError(
                "failed to configure DuckDB S3 access (httpfs)",
                details={"endpoint": s3.endpoint, "error": str(exc)},
            ) from exc
        self._s3_ready = True

    def execute(self, sql: str, params: list[Any] | None = None) -> pa.Table:
        if "s3://" in sql:
            self.ensure_s3()
        try:
            return self._con.execute(sql, params or []).to_arrow_table()
        except duckdb.Error as exc:
            raise LakehouseEngineError(
                "query failed", details={"sql": sql[:500], "error": str(exc)}
            ) from exc

    def execute_scalar(self, sql: str, params: list[Any] | None = None) -> Any:
        if "s3://" in sql:
            self.ensure_s3()
        try:
            row = self._con.execute(sql, params or []).fetchone()
        except duckdb.Error as exc:
            raise LakehouseEngineError(
                "query failed", details={"sql": sql[:500], "error": str(exc)}
            ) from exc
        return None if row is None else row[0]

    def register(self, name: str, table: pa.Table) -> None:
        self._con.register(name, table)

    def unregister(self, name: str) -> None:
        self._con.unregister(name)

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> DuckDBEngine:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
