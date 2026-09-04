"""Ingestion data contracts (PRD §3.2.1, §3.2.2)."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from akl import ids

SourceType = Literal["pdf", "markdown", "html", "github"]
SecurityLevel = Literal["public", "internal", "restricted"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Discovery / fetch
# ---------------------------------------------------------------------------
class SourceItem(_Frozen):
    """Something a connector found and may need to download."""

    uri: str
    canonical_uri: str
    source_type: SourceType
    filename: str | None = None
    expected_size: int | None = None
    hint_hash: str | None = None  # e.g. ETag, git blob sha — lets fetch() skip unchanged content
    security_level: SecurityLevel = "internal"
    allowed_groups: tuple[str, ...] = ()
    source_metadata: dict[str, str] = Field(default_factory=dict)


class FetchedObject(_Frozen):
    """Raw bytes plus provenance; input to Bronze and to parsers."""

    item: SourceItem
    data: bytes
    mime_type: str | None
    sha256: str
    size_bytes: int
    fetched_at: datetime
    source_metadata: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_bytes(
        cls,
        item: SourceItem,
        data: bytes,
        *,
        mime_type: str | None,
        source_metadata: dict[str, str] | None = None,
        fetched_at: datetime | None = None,
    ) -> FetchedObject:
        return cls(
            item=item,
            data=data,
            mime_type=mime_type,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            fetched_at=fetched_at or datetime.now(UTC),
            source_metadata={**item.source_metadata, **(source_metadata or {})},
        )

    @property
    def extension(self) -> str:
        name = self.item.filename or self.item.uri.rsplit("/", 1)[-1]
        return name.rsplit(".", 1)[-1].lower() if "." in name else ""


class DeletionEvent(_Frozen):
    """A source reports a document no longer exists (PRD §3.3.4)."""

    canonical_uri: str
    reason: str = "removed_at_source"


class ConnectorHealth(_Frozen):
    ok: bool
    latency_ms: float
    detail: str = ""


# ---------------------------------------------------------------------------
# Structural blocks (input to the chunking engine, PRD §3.2.2)
# ---------------------------------------------------------------------------
class _Block(_Frozen):
    start_char: int = 0
    end_char: int = 0


class HeadingBlock(_Block):
    kind: Literal["heading"] = "heading"
    level: int = Field(ge=1, le=6)
    text: str


class ParagraphBlock(_Block):
    kind: Literal["paragraph"] = "paragraph"
    text: str


class CodeBlock(_Block):
    kind: Literal["code"] = "code"
    text: str
    language: str | None = None
    symbol: str | None = None  # function/class name for code sources


class TableBlock(_Block):
    kind: Literal["table"] = "table"
    markdown: str
    n_rows: int
    n_cols: int
    caption: str | None = None


class ListBlock(_Block):
    kind: Literal["list"] = "list"
    items: tuple[str, ...]
    ordered: bool = False


class ImageBlock(_Block):
    kind: Literal["image"] = "image"
    alt: str | None = None
    width: int | None = None
    height: int | None = None
    page: int | None = None
    caption: str | None = None


class PageBreakBlock(_Block):
    kind: Literal["page_break"] = "page_break"
    page: int  # page number that STARTS after this break


Block = Annotated[
    HeadingBlock
    | ParagraphBlock
    | CodeBlock
    | TableBlock
    | ListBlock
    | ImageBlock
    | PageBreakBlock,
    Field(discriminator="kind"),
]


class HeadingNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: int
    text: str
    start_char: int
    end_char: int
    children: list[HeadingNode] = Field(default_factory=list)


class QualityReport(_Frozen):
    score: float = Field(ge=0.0, le=1.0)
    flags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Unified document
# ---------------------------------------------------------------------------
class UnifiedDocument(_Frozen):
    """Parser output: canonical text + structure + metadata (PRD §3.2.2)."""

    document_id: uuid.UUID
    content_sha256: str
    source_type: SourceType
    source_uri: str
    canonical_source_uri: str
    title: str | None
    language: str | None
    text: str
    blocks: tuple[Block, ...]
    structure: tuple[HeadingNode, ...] = ()
    page_map: tuple[dict[str, int], ...] = ()  # [{page, start_char, end_char}]
    quality: QualityReport = QualityReport(score=1.0)  # noqa: B008 - frozen, safe shared default
    fingerprint_simhash: int | None = None
    security_level: SecurityLevel = "internal"
    allowed_groups: tuple[str, ...] = ()
    metadata: dict[str, str] = Field(default_factory=dict)
    parser_name: str
    parser_version: str

    # -- derived indexes ----------------------------------------------------------
    @property
    def tables(self) -> list[dict[str, Any]]:
        return [
            {
                "id": i,
                "markdown": b.markdown,
                "n_rows": b.n_rows,
                "n_cols": b.n_cols,
                "start_char": b.start_char,
                "end_char": b.end_char,
            }
            for i, b in enumerate(self.blocks)
            if isinstance(b, TableBlock)
        ]

    @property
    def code_blocks(self) -> list[dict[str, Any]]:
        return [
            {
                "id": i,
                "language": b.language,
                "symbol": b.symbol,
                "start_char": b.start_char,
                "end_char": b.end_char,
            }
            for i, b in enumerate(self.blocks)
            if isinstance(b, CodeBlock)
        ]

    @property
    def images(self) -> list[dict[str, Any]]:
        return [
            {
                "id": i,
                "alt": b.alt,
                "width": b.width,
                "height": b.height,
                "page": b.page,
                "caption": b.caption,
            }
            for i, b in enumerate(self.blocks)
            if isinstance(b, ImageBlock)
        ]

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def document_version_id(self) -> uuid.UUID:
        return ids.document_version_id(self.document_id, self.content_sha256, self.parser_version)

    def to_silver_row(self, *, parsed_at: datetime | None = None) -> dict[str, Any]:
        """Row for ``silver/documents`` (PRD §2.5.1). Pass through ``SilverStore.prepare_document_row``."""
        parsed = parsed_at or datetime.now(UTC)
        return {
            "document_version_id": str(self.document_version_id),
            "document_id": str(self.document_id),
            "content_sha256": self.content_sha256,
            "source_type": self.source_type,
            "canonical_source_uri": self.canonical_source_uri,
            "source_uri": self.source_uri,
            "title": self.title,
            "language": self.language,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "structure": json.dumps([n.model_dump() for n in self.structure], ensure_ascii=False),
            "tables": json.dumps(self.tables, ensure_ascii=False),
            "code_blocks": json.dumps(self.code_blocks, ensure_ascii=False),
            "images": json.dumps(self.images, ensure_ascii=False),
            "page_map": json.dumps(list(self.page_map), ensure_ascii=False),
            "word_count": len(self.text.split()),
            "char_count": len(self.text),
            "quality_score": float(self.quality.score),
            "quality_flags": list(self.quality.flags),
            "fingerprint_simhash": self.fingerprint_simhash,
            "is_duplicate_of": None,
            "security_level": self.security_level,
            "allowed_groups": list(self.allowed_groups),
            "metadata": list(self.metadata.items()),
            "parser_name": self.parser_name,
            "parser_version": self.parser_version,
            "parsed_at": parsed,
            "is_current": True,
            "is_deleted": False,
            "ingest_date": parsed.date(),
        }
