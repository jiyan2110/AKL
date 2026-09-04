"""Deterministic identifiers and URI canonicalisation."""

from __future__ import annotations

import posixpath
import uuid
from urllib.parse import urlsplit, urlunsplit

AKL_NS_ROOT = uuid.uuid5(uuid.NAMESPACE_URL, "https://akl.internal/ns")
AKL_NS_DOC = uuid.uuid5(AKL_NS_ROOT, "document")
AKL_NS_CHUNK = uuid.uuid5(AKL_NS_ROOT, "chunk")
AKL_NS_MANIFEST = uuid.uuid5(AKL_NS_ROOT, "manifest")
AKL_NS_QA = uuid.uuid5(AKL_NS_ROOT, "qa")

_HTTP_SCHEMES = {"http", "https"}
_DEFAULT_PORTS = {"http": 80, "https": 443}


def canonicalize_uri(uri: str) -> str:
    """Normalize a source locator so equivalent references share an identity."""
    raw = uri.strip()
    if not raw:
        raise ValueError("uri must not be empty")
    parts = urlsplit(raw)
    scheme = parts.scheme.lower()
    if not scheme:
        raise ValueError(f"uri must include a scheme: {uri!r}")
    host = (parts.hostname or "").lower()
    port = parts.port
    if port is not None and _DEFAULT_PORTS.get(scheme) != port:
        host = f"{host}:{port}"
    path = posixpath.normpath(parts.path) if parts.path else ""
    if path == ".":
        path = ""
    if scheme in _HTTP_SCHEMES:
        return urlunsplit((scheme, host, path.rstrip("/") or "/", "", ""))
    if scheme == "github":
        segments = [segment for segment in parts.path.split("/") if segment]
        if len(segments) < 2:
            raise ValueError(f"github uri must be github://owner/repo/branch/path: {uri!r}")
        return f"github://{host}/{'/'.join(segments)}"
    if scheme == "file":
        return urlunsplit((scheme, "", path or "/", "", ""))
    return urlunsplit((scheme, host, path, parts.query, ""))


def document_id(canonical_source_uri: str) -> uuid.UUID:
    return uuid.uuid5(AKL_NS_DOC, canonical_source_uri)


def manifest_id(doc_id: uuid.UUID, content_sha256: str, run_id: str) -> uuid.UUID:
    return uuid.uuid5(AKL_NS_MANIFEST, f"{doc_id}:{content_sha256}:{run_id}")


def document_version_id(doc_id: uuid.UUID, content_sha256: str, parser_version: str) -> uuid.UUID:
    return uuid.uuid5(AKL_NS_DOC, f"{doc_id}:{content_sha256}:{parser_version}")
