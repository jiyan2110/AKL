"""Connector contract and runner (PRD §3.2.1, §3.3, §3.7)."""

from __future__ import annotations

import fnmatch
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field

from akl.errors import AKLError
from akl.ingestion.models import (
    ConnectorHealth,
    DeletionEvent,
    FetchedObject,
    SecurityLevel,
    SourceItem,
    SourceType,
)


class ConnectorError(AKLError):
    """Source unreachable or fetch failed after retries (AKL-E3020)."""

    code = "AKL-E3020"
    retryable = True


class ConnectorConfigError(AKLError):
    """Invalid connector YAML (AKL-E0002 family)."""

    code = "AKL-E0002"
    retryable = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class PathRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    glob: str
    security_level: SecurityLevel | None = None
    allowed_groups: list[str] | None = None


class ConnectorConfig(BaseModel):
    """Base config for a connector *instance* (one YAML file under configs/connectors/)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,62}$")
    type: SourceType
    enabled: bool = True
    security_level: SecurityLevel = "internal"
    allowed_groups: list[str] = Field(default_factory=list)
    owners: list[str] = Field(default_factory=list)
    path_rules: list[PathRule] = Field(default_factory=list)
    fetch_concurrency: int = Field(default=8, ge=1, le=64)
    max_items_per_run: int | None = Field(default=None, ge=1)

    def resolve_security(self, path: str) -> tuple[SecurityLevel, tuple[str, ...]]:
        """First matching path rule wins; rules may RAISE but never lower the level (PRD §9.3)."""
        level: SecurityLevel = self.security_level
        groups: tuple[str, ...] = tuple(self.allowed_groups)
        order = {"public": 0, "internal": 1, "restricted": 2}
        for rule in self.path_rules:
            if fnmatch.fnmatch(path, rule.glob):
                if rule.security_level and order[rule.security_level] > order[level]:
                    level = rule.security_level
                if rule.allowed_groups is not None:
                    groups = tuple(rule.allowed_groups)
                break
        return level, groups


def load_connector_configs(
    directory: Path, config_types: Mapping[str, type[ConnectorConfig]]
) -> list[ConnectorConfig]:
    """Load every ``*.yaml`` under ``directory`` into its typed config by ``type``."""
    configs: list[ConnectorConfig] = []
    for path in sorted(directory.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict) or "type" not in raw:
            raise ConnectorConfigError(f"{path.name}: missing 'type'", details={"file": str(path)})
        cls = config_types.get(str(raw["type"]))
        if cls is None:
            raise ConnectorConfigError(
                f"{path.name}: unknown connector type {raw['type']!r}", details={"file": str(path)}
            )
        try:
            configs.append(cls.model_validate(raw))
        except ValueError as exc:
            raise ConnectorConfigError(f"{path.name}: {exc}", details={"file": str(path)}) from exc
    return configs


# ---------------------------------------------------------------------------
# Connector contract
# ---------------------------------------------------------------------------
class BaseConnector(ABC):
    """Discover → fetch → checkpoint (PRD §3.2.1). Subclasses set ``name``, ``version``, ``source_type``."""

    name: str = "base"
    version: str = "0.0.0"
    source_type: SourceType

    retry_attempts: int = 5
    retry_base_seconds: float = 1.0
    retry_max_seconds: float = 32.0

    def __init__(self, config: ConnectorConfig) -> None:
        self.config = config

    @abstractmethod
    def discover(self, state: Mapping[str, Any]) -> Iterator[SourceItem | DeletionEvent]:
        """Yield changed items (and deletions) since ``state``. MUST NOT download bodies."""

    @abstractmethod
    def fetch(self, item: SourceItem) -> FetchedObject:
        """Download one item. Raise :class:`ConnectorError` on failure."""

    @abstractmethod
    def checkpoint(
        self, state: Mapping[str, Any], committed: Sequence[FetchedObject]
    ) -> dict[str, Any]:
        """Return the new state after ``committed`` items were durably stored."""

    def health(self) -> ConnectorHealth:
        return ConnectorHealth(ok=True, latency_ms=0.0, detail="not implemented")

    def fetch_with_retry(self, item: SourceItem, *, sleep: Any = time.sleep) -> FetchedObject:
        """Exponential backoff wrapper: 1s, 2s, 4s, 8s, 16s (capped at ``retry_max_seconds``)."""
        last: Exception | None = None
        for attempt in range(self.retry_attempts):
            try:
                return self.fetch(item)
            except ConnectorError as exc:
                last = exc
                if not exc.retryable or attempt == self.retry_attempts - 1:
                    break
                sleep(min(self.retry_base_seconds * (2**attempt), self.retry_max_seconds))
        raise ConnectorError(
            f"fetch failed after {self.retry_attempts} attempts: {item.uri}",
            details={"uri": item.uri, "error": str(last)},
        )


# ---------------------------------------------------------------------------
# Runner (Bronze write flow, PRD §3.7)
# ---------------------------------------------------------------------------
class RawPutLike(Protocol):
    @property
    def object_key(self) -> str: ...

    @property
    def content_sha256(self) -> str: ...

    @property
    def size_bytes(self) -> int: ...

    @property
    def deduplicated(self) -> bool: ...


class BronzeSink(Protocol):
    def put_raw(
        self,
        data: bytes,
        *,
        source_type: str,
        mime_type: str | None = None,
        filename: str | None = None,
    ) -> RawPutLike: ...

    @staticmethod
    def build_manifest_row(
        *,
        source_uri: str,
        source_type: str,
        put: Any,
        connector_name: str,
        connector_version: str,
        run_id: str,
        mime_type: str | None,
        fetched_at: datetime | None = None,
        source_metadata: Mapping[str, str] | None = None,
    ) -> dict[str, Any]: ...

    def write_manifest(self, rows: Sequence[Mapping[str, Any]], *, run_id: str) -> Any: ...


class DocumentRecorder(Protocol):
    def record_bronze(
        self,
        *,
        canonical_source_uri: str,
        source_type: str,
        connector_id: str,
        content_sha256: str,
        bronze_object_key: str,
        run_id: str,
        fetched_at: datetime | None = None,
        security_level: str = "internal",
        allowed_groups: Sequence[str] = (),
        title: str | None = None,
        parser_version: str = "",
    ) -> Any: ...


@dataclass
class IngestionRunReport:
    connector_id: str
    run_id: str
    discovered: int = 0
    fetched: int = 0
    deduplicated: int = 0
    failed: int = 0
    bytes_fetched: int = 0
    manifest_rows: int = 0
    deletions: list[DeletionEvent] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)
    committed: list[FetchedObject] = field(default_factory=list)
    new_state: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0


class ConnectorRunner:
    """Executes one connector run against Bronze and the metadata catalogue."""

    def __init__(self, bronze: BronzeSink, documents: DocumentRecorder | None = None) -> None:
        self._bronze = bronze
        self._documents = documents

    def run(
        self,
        connector: BaseConnector,
        *,
        state: Mapping[str, Any],
        run_id: str,
        items: Iterable[SourceItem | DeletionEvent] | None = None,
    ) -> IngestionRunReport:
        start = time.perf_counter()
        cfg = connector.config
        report = IngestionRunReport(connector_id=cfg.id, run_id=run_id)
        manifest_rows: list[dict[str, Any]] = []

        to_fetch: list[SourceItem] = []
        for event in items if items is not None else connector.discover(state):
            if isinstance(event, DeletionEvent):
                report.deletions.append(event)
                continue
            report.discovered += 1
            if cfg.max_items_per_run is None or len(to_fetch) < cfg.max_items_per_run:
                to_fetch.append(event)

        # Fetch concurrently but COMMIT in discovery order so Bronze/Postgres/manifest
        # ordering is deterministic across runs (matters for dedup tie-breaks and audits).
        with ThreadPoolExecutor(max_workers=cfg.fetch_concurrency) as pool:
            futures = [(item, pool.submit(connector.fetch_with_retry, item)) for item in to_fetch]
            for item, future in futures:
                try:
                    fetched = future.result()
                except AKLError as exc:
                    report.failed += 1
                    report.failures.append(
                        {"uri": item.uri, "code": exc.code, "error": exc.message}
                    )
                    continue
                self._commit(connector, fetched, run_id, report, manifest_rows)

        if manifest_rows:
            self._bronze.write_manifest(manifest_rows, run_id=run_id)
            report.manifest_rows = len(manifest_rows)
        report.new_state = connector.checkpoint(state, report.committed)
        report.duration_s = time.perf_counter() - start
        return report

    def _commit(
        self,
        connector: BaseConnector,
        fetched: FetchedObject,
        run_id: str,
        report: IngestionRunReport,
        manifest_rows: list[dict[str, Any]],
    ) -> None:
        item = fetched.item
        put = self._bronze.put_raw(
            fetched.data,
            source_type=item.source_type,
            mime_type=fetched.mime_type,
            filename=item.filename,
        )
        if self._documents is not None:
            self._documents.record_bronze(
                canonical_source_uri=item.canonical_uri,
                source_type=item.source_type,
                connector_id=connector.config.id,
                content_sha256=put.content_sha256,
                bronze_object_key=put.object_key,
                run_id=run_id,
                fetched_at=fetched.fetched_at,
                security_level=item.security_level,
                allowed_groups=item.allowed_groups,
            )
        manifest_rows.append(
            self._bronze.build_manifest_row(
                source_uri=item.uri,
                source_type=item.source_type,
                put=put,
                connector_name=connector.name,
                connector_version=connector.version,
                run_id=run_id,
                mime_type=fetched.mime_type,
                fetched_at=fetched.fetched_at,
                source_metadata=fetched.source_metadata,
            )
        )
        report.fetched += 1
        report.bytes_fetched += fetched.size_bytes
        if put.deduplicated:
            report.deduplicated += 1
        report.committed.append(fetched)
