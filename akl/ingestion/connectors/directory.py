"""Directory-based connector shared by the Markdown and PDF sources (PRD §3.3.1, §3.3.2).

Discovery compares ``(mtime, size)`` per file against the checkpoint; files that
disappeared since the last run become :class:`DeletionEvent`s.
"""

from __future__ import annotations

import fnmatch
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from akl import ids
from akl.ingestion.connectors.base import BaseConnector, ConnectorConfig, ConnectorError
from akl.ingestion.models import (
    ConnectorHealth,
    DeletionEvent,
    FetchedObject,
    SourceItem,
    SourceType,
)


class DirectoryConnectorConfig(ConnectorConfig):
    """Config for connectors that watch a local directory (or mounted volume)."""

    root_path: Path
    include_globs: list[str] = Field(default_factory=lambda: ["**/*"])
    exclude_globs: list[str] = Field(default_factory=lambda: ["**/node_modules/**", "**/.git/**"])
    uri_base: str | None = (
        None  # e.g. https://docs.example.internal/runbooks ; default file://<root>
    )


def glob_match(rel_posix: str, pattern: str) -> bool:
    """fnmatch with ``**/`` prefix tolerance so patterns match at any depth including root."""
    if fnmatch.fnmatch(rel_posix, pattern):
        return True
    return pattern.startswith("**/") and fnmatch.fnmatch(rel_posix, pattern[3:])


class DirectoryConnector(BaseConnector):
    """Generic directory watcher. Subclasses set ``source_type``, ``mime_type`` and ``name``."""

    mime_type: ClassVar[str] = "application/octet-stream"
    config_cls: ClassVar[type[ConnectorConfig]] = DirectoryConnectorConfig
    source_type: SourceType = "markdown"

    def __init__(self, config: ConnectorConfig) -> None:
        if not isinstance(config, DirectoryConnectorConfig):
            config = self.config_cls.model_validate(config.model_dump())
        super().__init__(config)
        self.cfg: DirectoryConnectorConfig = config  # type: ignore[assignment]
        self._missing: set[str] = set()

    # -- helpers -------------------------------------------------------------------
    @property
    def root(self) -> Path:
        return self.cfg.root_path.resolve()

    def uri_for(self, rel: str) -> str:
        base = self.cfg.uri_base or self.root.as_uri()
        return f"{base.rstrip('/')}/{rel}"

    def matches(self, rel: str) -> bool:
        if any(glob_match(rel, p) for p in self.cfg.exclude_globs):
            return False
        return any(glob_match(rel, p) for p in self.cfg.include_globs)

    def scan(self) -> dict[str, dict[str, float | int]]:
        """Current files under root: ``{relpath: {"mtime": float, "size": int}}``."""
        out: dict[str, dict[str, float | int]] = {}
        if not self.root.is_dir():
            return out
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(self.root).as_posix()
            if self.matches(rel):
                st = path.stat()
                out[rel] = {"mtime": round(st.st_mtime, 3), "size": st.st_size}
        return out

    # -- contract -------------------------------------------------------------------
    def discover(self, state: Mapping[str, Any]) -> Iterator[SourceItem | DeletionEvent]:
        known: dict[str, Any] = dict(state.get("files", {}))
        current = self.scan()
        for rel, meta in sorted(current.items()):
            if known.get(rel) == meta:
                continue
            level, groups = self.cfg.resolve_security(rel)
            uri = self.uri_for(rel)
            yield SourceItem(
                uri=uri,
                canonical_uri=ids.canonicalize_uri(uri),
                source_type=self.source_type,
                filename=Path(rel).name,
                expected_size=int(meta["size"]),
                security_level=level,
                allowed_groups=groups,
                source_metadata={"relpath": rel, "mtime": str(meta["mtime"])},
            )
        self._missing = set(known) - set(current)
        for rel in sorted(self._missing):
            yield DeletionEvent(canonical_uri=ids.canonicalize_uri(self.uri_for(rel)))

    def fetch(self, item: SourceItem) -> FetchedObject:
        rel = item.source_metadata.get("relpath", "")
        path = self.root / rel
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise ConnectorError(
                f"file vanished: {rel}", details={"path": str(path)}, retryable=False
            ) from exc
        except OSError as exc:
            raise ConnectorError(
                f"read failed: {rel}", details={"path": str(path), "error": str(exc)}
            ) from exc
        return FetchedObject.from_bytes(
            item, data, mime_type=self.mime_type, source_metadata={"filename": path.name}
        )

    def checkpoint(
        self, state: Mapping[str, Any], committed: Sequence[FetchedObject]
    ) -> dict[str, Any]:
        files: dict[str, Any] = dict(state.get("files", {}))
        for obj in committed:
            rel = obj.item.source_metadata["relpath"]
            files[rel] = {"mtime": float(obj.item.source_metadata["mtime"]), "size": obj.size_bytes}
        for rel in self._missing:
            files.pop(rel, None)
        return {"files": files, "last_scan": time.time()}

    def health(self) -> ConnectorHealth:
        start = time.perf_counter()
        ok = self.root.is_dir() and os.access(self.root, os.R_OK)
        return ConnectorHealth(
            ok=ok, latency_ms=(time.perf_counter() - start) * 1000, detail=str(self.root)
        )
