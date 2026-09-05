"""Chunk data model (PRD §2.5.2, §4.8)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

ChunkType = str  # prose | code | table | heading_only | list | mixed


class ChunkStatus(StrEnum):
    UNCHANGED = "unchanged"
    MODIFIED = "modified"
    MOVED = "moved"
    ADDED = "added"
    REMOVED = "removed"


@dataclass
class Chunk:
    """One retrieval unit produced by the engine; ``to_silver_row`` renders the Silver row."""

    document_id: uuid.UUID
    document_version_id: uuid.UUID
    source_type: str
    chunk_index: int
    chunk_type: ChunkType
    heading_path: tuple[str, ...]
    heading_level: int | None
    text: str
    context_prefix: str
    start_char: int
    end_char: int
    token_count: int
    overlap_prev_tokens: int = 0
    page_start: int | None = None
    page_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    language: str | None = None
    code_language: str | None = None
    quality_score: float = 1.0
    quality_flags: tuple[str, ...] = ()
    security_level: str = "internal"
    allowed_groups: tuple[str, ...] = ()
    chunker_version: str = "0.0.0"
    chunk_config_hash: str = ""
    section_ordinal: int = 0  # ordinal within its section → chunk_key
    # identity (filled by identity.assign_identity)
    chunk_key: str = ""
    chunk_checksum: str = ""
    embedded_text_sha256: str = ""
    chunk_id: uuid.UUID = field(default_factory=lambda: uuid.UUID(int=0))
    lineage_id: uuid.UUID = field(default_factory=lambda: uuid.UUID(int=0))
    prev_chunk_id: uuid.UUID | None = None
    next_chunk_id: uuid.UUID | None = None
    parent_chunk_id: uuid.UUID | None = None
    status: ChunkStatus = ChunkStatus.ADDED

    @property
    def embedded_text(self) -> str:
        return f"{self.context_prefix}\n{self.text}" if self.context_prefix else self.text

    def to_silver_row(self, *, created_at: datetime | None = None) -> dict[str, Any]:
        now = created_at or datetime.now(UTC)
        return {
            "chunk_id": str(self.chunk_id),
            "chunk_key": self.chunk_key,
            "lineage_id": str(self.lineage_id),
            "chunk_checksum": self.chunk_checksum,
            "embedded_text_sha256": self.embedded_text_sha256,
            "document_id": str(self.document_id),
            "document_version_id": str(self.document_version_id),
            "chunk_index": self.chunk_index,
            "chunk_type": self.chunk_type,
            "heading_path": list(self.heading_path),
            "heading_level": self.heading_level,
            "text": self.text,
            "context_prefix": self.context_prefix,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "token_count": self.token_count,
            "overlap_prev_tokens": self.overlap_prev_tokens,
            "language": self.language,
            "code_language": self.code_language,
            "quality_score": float(self.quality_score),
            "quality_flags": list(self.quality_flags),
            "prev_chunk_id": str(self.prev_chunk_id) if self.prev_chunk_id else None,
            "next_chunk_id": str(self.next_chunk_id) if self.next_chunk_id else None,
            "parent_chunk_id": str(self.parent_chunk_id) if self.parent_chunk_id else None,
            "chunker_version": self.chunker_version,
            "chunk_config_hash": self.chunk_config_hash,
            "security_level": self.security_level,
            "allowed_groups": list(self.allowed_groups),
            "source_type": self.source_type,
            "created_at": now,
            "is_current": True,
            "is_deleted": False,
            "ingest_date": now.date(),
        }


@dataclass
class ChunkDiff:
    """Result of the incremental update algorithm (PRD §4.12)."""

    document_id: uuid.UUID
    unchanged: int = 0
    modified: int = 0
    moved: int = 0
    added: int = 0
    removed: int = 0
    to_write: list[Chunk] = field(default_factory=list)
    removed_ids: list[uuid.UUID] = field(default_factory=list)
    superseded_ids: list[uuid.UUID] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.to_write or self.removed_ids)
