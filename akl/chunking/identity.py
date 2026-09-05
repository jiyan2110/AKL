"""Chunk identity, context prefix and lineage (PRD §4.8, §4.10, ADR-003)."""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from collections.abc import Sequence

from akl import ids

_WS = re.compile(r"\s+")
BREADCRUMB_SEP = " › "
MAX_PATH_LEVELS = 4


def normalized_text(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def chunk_checksum(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


def chunk_key(document_id: uuid.UUID, heading_path: Sequence[str], ordinal: int) -> str:
    raw = f"{document_id}\x1f{'/'.join(heading_path)}\x1f{ordinal}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:40]  # noqa: S324 - identity, not security


def chunk_id(document_id: uuid.UUID, key: str, checksum: str) -> uuid.UUID:
    return uuid.uuid5(ids.AKL_NS_CHUNK, f"{document_id}:{key}:{checksum}")


def embedded_text_sha256(context_prefix: str, text: str) -> str:
    embedded = f"{context_prefix}\n{text}" if context_prefix else text
    return hashlib.sha256(embedded.encode("utf-8")).hexdigest()


def render_context_prefix(
    title: str | None, heading_path: Sequence[str], *, max_tokens: int, count: object
) -> str:
    """``<title> › <h1> › <h2> …`` truncated to the last ``MAX_PATH_LEVELS`` headings and ``max_tokens``."""
    parts = [p for p in [title, *heading_path[-MAX_PATH_LEVELS:]] if p]
    # drop title if it duplicates the first heading
    if len(parts) >= 2 and parts[0].strip().lower() == parts[1].strip().lower():
        parts = parts[1:]
    prefix = BREADCRUMB_SEP.join(parts)
    counter = count  # Callable[[str], int]
    while parts and counter(prefix) > max_tokens:  # type: ignore[operator]
        parts = parts[1:]
        prefix = BREADCRUMB_SEP.join(parts)
    return prefix
