"""Parser contract and the shared document assembler (PRD §3.4, §3.2.2)."""

from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from akl import ids
from akl.errors import AKLError
from akl.ingestion.models import (
    Block,
    CodeBlock,
    FetchedObject,
    HeadingBlock,
    HeadingNode,
    ImageBlock,
    ListBlock,
    PageBreakBlock,
    ParagraphBlock,
    QualityReport,
    SecurityLevel,
    SourceType,
    TableBlock,
    UnifiedDocument,
)


class ParseError(AKLError):
    """Parser failed on an input (AKL-E3030)."""

    code = "AKL-E3030"
    retryable = False


class UnsupportedFormatError(AKLError):
    """No parser accepts this MIME/extension (AKL-E3003)."""

    code = "AKL-E3003"
    http_status = 415
    retryable = False


class BaseParser(ABC):
    """Bytes → :class:`UnifiedDocument`. Subclasses set ``name``, ``version``."""

    name: str = "base"
    version: str = "0.0.0"
    mime_types: tuple[str, ...] = ()
    extensions: tuple[str, ...] = ()
    source_types: tuple[SourceType, ...] = ()

    def supports(self, mime_type: str | None, extension: str, source_type: str) -> bool:
        mime = (mime_type or "").split(";")[0].strip().lower()
        if self.source_types and source_type not in self.source_types:
            return False
        return mime in self.mime_types or extension.lower() in self.extensions

    @abstractmethod
    def parse(self, obj: FetchedObject) -> UnifiedDocument:
        """Produce a UnifiedDocument; raise :class:`ParseError` on failure."""


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------
_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")


def normalize_text(text: str) -> str:
    """NFC, LF line endings, no trailing whitespace, at most two consecutive newlines."""
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = _TRAILING_WS.sub("\n", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


class DocumentAssembler:
    """Collects blocks in reading order and renders canonical text with exact offsets."""

    def __init__(self) -> None:
        self._pending: list[tuple[Any, str]] = []  # (block without offsets, rendered text)
        self._current_page = 1

    # -- adding blocks (offsets assigned at build) ---------------------------------
    def add_heading(self, level: int, text: str) -> None:
        text = normalize_text(text).replace("\n", " ")
        lvl = max(1, min(level, 6))
        if text:
            self._pending.append((HeadingBlock(level=lvl, text=text), f"{'#' * lvl} {text}"))

    def add_paragraph(self, text: str) -> None:
        text = normalize_text(text)
        if text:
            self._pending.append((ParagraphBlock(text=text), text))

    def add_code(self, text: str, language: str | None = None, symbol: str | None = None) -> None:
        body = text.replace("\r\n", "\n").rstrip()
        if body.strip():
            fence = f"```{language or ''}".rstrip()
            self._pending.append(
                (CodeBlock(text=body, language=language, symbol=symbol), f"{fence}\n{body}\n```")
            )

    def add_table(
        self, markdown: str, n_rows: int, n_cols: int, caption: str | None = None
    ) -> None:
        md = normalize_text(markdown)
        if md:
            rendered = f"{caption}\n\n{md}" if caption else md
            self._pending.append(
                (TableBlock(markdown=md, n_rows=n_rows, n_cols=n_cols, caption=caption), rendered)
            )

    def add_list(self, items: Sequence[str], ordered: bool = False) -> None:
        clean = tuple(normalize_text(i).replace("\n", " ") for i in items if normalize_text(i))
        if clean:
            rendered = "\n".join(
                f"{i + 1}. {t}" if ordered else f"- {t}" for i, t in enumerate(clean)
            )
            self._pending.append((ListBlock(items=clean, ordered=ordered), rendered))

    def add_image(
        self,
        alt: str | None = None,
        *,
        width: int | None = None,
        height: int | None = None,
        page: int | None = None,
        caption: str | None = None,
    ) -> None:
        block = ImageBlock(
            alt=alt, width=width, height=height, page=page or self._current_page, caption=caption
        )
        self._pending.append((block, f"![{alt or 'image'}]"))

    def page_break(self) -> None:
        """Mark that the following blocks belong to the next page (PDF)."""
        self._current_page += 1
        self._pending.append((PageBreakBlock(page=self._current_page), ""))

    # -- build --------------------------------------------------------------------
    def build(
        self,
        *,
        source_uri: str,
        source_type: SourceType,
        content_sha256: str,
        parser_name: str,
        parser_version: str,
        title: str | None = None,
        language: str | None = None,
        security_level: SecurityLevel = "internal",
        allowed_groups: Sequence[str] = (),
        metadata: dict[str, str] | None = None,
        quality: QualityReport | None = None,
        fingerprint_simhash: int | None = None,
    ) -> UnifiedDocument:
        text_parts: list[str] = []
        blocks: list[Block] = []
        page_map: list[dict[str, int]] = []
        cursor = 0
        page_start = 0
        page_no = 1

        for block, rendered in self._pending:
            if isinstance(block, PageBreakBlock):
                page_map.append({"page": page_no, "start_char": page_start, "end_char": cursor})
                blocks.append(block.model_copy(update={"start_char": cursor, "end_char": cursor}))
                page_no = block.page
                page_start = cursor
                continue
            if text_parts:
                text_parts.append("\n\n")
                cursor += 2
            start = cursor
            text_parts.append(rendered)
            cursor += len(rendered)
            blocks.append(block.model_copy(update={"start_char": start, "end_char": cursor}))
        if page_no > 1 or page_map:
            page_map.append({"page": page_no, "start_char": page_start, "end_char": cursor})

        text = "".join(text_parts)
        canonical = ids.canonicalize_uri(source_uri)
        inferred_title = title or next(
            (b.text for b in blocks if isinstance(b, HeadingBlock)), None
        )
        return UnifiedDocument(
            document_id=ids.document_id(canonical),
            content_sha256=content_sha256,
            source_type=source_type,
            source_uri=source_uri,
            canonical_source_uri=canonical,
            title=inferred_title,
            language=language,
            text=text,
            blocks=tuple(blocks),
            structure=tuple(build_heading_tree(blocks, len(text))),
            page_map=tuple(page_map),
            quality=quality or QualityReport(score=1.0),
            fingerprint_simhash=fingerprint_simhash,
            security_level=security_level,
            allowed_groups=tuple(allowed_groups),
            metadata=metadata or {},
            parser_name=parser_name,
            parser_version=parser_version,
        )


def build_heading_tree(blocks: Sequence[Block], text_length: int) -> list[HeadingNode]:
    """Nest headings by level; each node spans from its heading to the next heading of equal/higher level."""
    headings = [b for b in blocks if isinstance(b, HeadingBlock)]
    nodes: list[HeadingNode] = []
    for i, h in enumerate(headings):
        end = text_length
        for later in headings[i + 1 :]:
            if later.level <= h.level:
                end = later.start_char
                break
        nodes.append(HeadingNode(level=h.level, text=h.text, start_char=h.start_char, end_char=end))

    roots: list[HeadingNode] = []
    stack: list[HeadingNode] = []
    for node in nodes:
        while stack and stack[-1].level >= node.level:
            stack.pop()
        (stack[-1].children if stack else roots).append(node)
        stack.append(node)
    return roots
