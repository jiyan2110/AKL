"""PDF parser (PRD §3.4.2) built on PyMuPDF.

Strategy per page: text blocks with font sizes → heading detection by font-size
rank; repeated header/footer bands removed; tables via ``page.find_tables()``;
monospace runs → code; image-only pages flagged. OCR is a non-goal (PRD §0.4).
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pymupdf

from akl.errors import AKLError
from akl.ingestion.models import FetchedObject, QualityReport, UnifiedDocument
from akl.ingestion.parsers.base import BaseParser, DocumentAssembler, ParseError

MAX_PAGES_DEFAULT = 2000
_MONO_HINTS = ("mono", "courier", "consolas", "menlo", "code")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


class PdfEncryptedError(AKLError):
    code = "AKL-E3010"
    retryable = False


class PdfCorruptError(AKLError):
    code = "AKL-E3011"
    retryable = False


class PdfTooLargeError(AKLError):
    code = "AKL-E3012"
    retryable = False


class PdfNoTextError(AKLError):
    code = "AKL-E3013"
    retryable = False


@dataclass
class _Span:
    text: str
    size: float
    bold: bool
    mono: bool
    y: float
    x: float


@dataclass
class _Line:
    spans: list[_Span]

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans).strip()

    @property
    def size(self) -> float:
        return max((s.size for s in self.spans), default=0.0)

    @property
    def bold(self) -> bool:
        return bool(self.spans) and all(s.bold for s in self.spans if s.text.strip())

    @property
    def mono(self) -> bool:
        return bool(self.spans) and all(s.mono for s in self.spans if s.text.strip())

    @property
    def y(self) -> float:
        return min((s.y for s in self.spans), default=0.0)


def _lines_from_page(page: Any) -> tuple[list[_Line], int]:
    """Extract lines (with font metadata) in reading order; returns (lines, image_count)."""
    data = page.get_text("dict")
    lines: list[_Line] = []
    images = 0
    for block in data.get("blocks", []):
        if block.get("type") == 1:
            images += 1
            continue
        for line in block.get("lines", []):
            spans: list[_Span] = []
            for sp in line.get("spans", []):
                text = sp.get("text", "")
                if not text:
                    continue
                font = str(sp.get("font", "")).lower()
                flags = int(sp.get("flags", 0))
                spans.append(
                    _Span(
                        text=text,
                        size=float(sp.get("size", 0.0)),
                        bold=bool(flags & 16) or "bold" in font,
                        mono=any(h in font for h in _MONO_HINTS) or bool(flags & 8),
                        y=float(sp["bbox"][1]),
                        x=float(sp["bbox"][0]),
                    )
                )
            if spans and "".join(s.text for s in spans).strip():
                lines.append(_Line(spans))
    lines.sort(key=lambda ln: (round(ln.y / 3), ln.spans[0].x))
    return lines, images


def _boilerplate_lines(pages: list[list[_Line]], page_height: float) -> set[str]:
    """Lines repeated in the top/bottom 8% band on ≥60% of pages are headers/footers."""
    if len(pages) < 3:
        return set()
    counter: Counter[str] = Counter()
    for lines in pages:
        seen: set[str] = set()
        for ln in lines:
            band = ln.y < page_height * 0.08 or ln.y > page_height * 0.92
            norm = re.sub(r"\d+", "#", ln.text)
            if band and norm and norm not in seen:
                seen.add(norm)
                counter[norm] += 1
    threshold = max(2, int(0.6 * len(pages)))
    return {t for t, n in counter.items() if n >= threshold}


class PdfParser(BaseParser):
    name = "pdf"
    version = "1.0.0"
    mime_types = ("application/pdf",)
    extensions = ("pdf",)
    source_types = ("pdf",)

    def __init__(
        self, *, max_pages: int = MAX_PAGES_DEFAULT, try_empty_password: bool = True
    ) -> None:
        self.max_pages = max_pages
        self.try_empty_password = try_empty_password

    def parse(self, obj: FetchedObject) -> UnifiedDocument:
        try:
            doc = pymupdf.open(stream=obj.data, filetype="pdf")
        except Exception as exc:
            raise PdfCorruptError("cannot open PDF", details={"error": str(exc)}) from exc
        try:
            if doc.is_encrypted:
                if not (self.try_empty_password and doc.authenticate("")):
                    raise PdfEncryptedError("PDF is encrypted")
            if doc.page_count > self.max_pages:
                raise PdfTooLargeError(f"PDF has {doc.page_count} pages > {self.max_pages}")
            return self._parse_open(doc, obj)
        except AKLError:
            raise
        except Exception as exc:
            raise ParseError("PDF parsing failed", details={"error": str(exc)}) from exc
        finally:
            doc.close()

    # -- core -------------------------------------------------------------------------
    def _parse_open(self, doc: Any, obj: FetchedObject) -> UnifiedDocument:
        pages: list[list[_Line]] = []
        image_counts: list[int] = []
        page_height = float(doc[0].rect.height) if doc.page_count else 800.0
        tables_by_page: dict[int, list[tuple[Any, str, int, int]]] = {}
        for pno, page in enumerate(doc):
            lines, images = _lines_from_page(page)
            pages.append(lines)
            image_counts.append(images)
            tables_by_page[pno] = self._tables(page)

        all_sizes = [ln.size for lines in pages for ln in lines if ln.text]
        if not all_sizes:
            raise PdfNoTextError("no extractable text (image-only PDF)")
        body_size = statistics.mode(round(s, 1) for s in all_sizes)
        heading_sizes = sorted(
            {round(s, 1) for s in all_sizes if s >= body_size * 1.15}, reverse=True
        )[:4]
        boiler = _boilerplate_lines(pages, page_height)

        asm = DocumentAssembler()
        flags: list[str] = []
        image_only_pages = 0
        for pno, lines in enumerate(pages):
            if pno > 0:
                asm.page_break()
            if not lines and image_counts[pno]:
                image_only_pages += 1
                asm.add_image(alt=None, page=pno + 1)
                continue
            _PageEmitter(asm, body_size, heading_sizes, boiler, tables_by_page.get(pno, [])).emit(
                lines
            )

        if image_only_pages:
            flags.append("image_only_pages")
        if any(count > 0 for count in image_counts):
            pass  # images recorded via add_image on image-only pages only; metadata below
        meta = {
            "pdf.pages": str(doc.page_count),
            "pdf.author": str(doc.metadata.get("author") or ""),
            "pdf.title": str(doc.metadata.get("title") or ""),
            "pdf.images": str(sum(image_counts)),
        }
        meta.update(obj.source_metadata)
        title = meta["pdf.title"] or None
        return asm.build(
            source_uri=obj.item.uri,
            source_type=obj.item.source_type,
            content_sha256=obj.sha256,
            parser_name=self.name,
            parser_version=self.version,
            title=title,
            security_level=obj.item.security_level,
            allowed_groups=obj.item.allowed_groups,
            metadata=meta,
            quality=QualityReport(score=1.0, flags=tuple(flags)),
        )

    @staticmethod
    def _heading_level(line: _Line, body_size: float, heading_sizes: list[float]) -> int:
        text = line.text
        if len(text) > 120 or text.endswith(".") or not heading_sizes:
            return 0
        size = round(line.size, 1)
        if size in heading_sizes:
            return heading_sizes.index(size) + 1
        if line.bold and size >= body_size and len(text) < 80:
            return min(len(heading_sizes) + 1, 6)
        return 0

    @staticmethod
    def _tables(page: Any) -> list[tuple[Any, str, int, int]]:
        out: list[tuple[Any, str, int, int]] = []
        try:
            found = page.find_tables()
        except Exception:  # find_tables can fail on exotic pages; tables are best-effort
            return out
        for tbl in found.tables:
            rows = tbl.extract()
            rows = [
                [(c or "").replace("\n", " ").strip() for c in r] for r in rows if any(c for c in r)
            ]
            if len(rows) < 2:
                continue
            n_cols = max(len(r) for r in rows)
            rows = [r + [""] * (n_cols - len(r)) for r in rows]
            md = "| " + " | ".join(rows[0]) + " |\n|" + "---|" * n_cols
            md += "".join("\n| " + " | ".join(r) + " |" for r in rows[1:])
            out.append((tbl.bbox, md, len(rows) - 1, n_cols))
        return out


class _PageEmitter:
    """Turns one page's lines into assembler blocks (paragraph/code/heading/table)."""

    def __init__(
        self,
        asm: DocumentAssembler,
        body_size: float,
        heading_sizes: list[float],
        boiler: set[str],
        tables: list[tuple[Any, str, int, int]],
    ) -> None:
        self.asm = asm
        self.body_size = body_size
        self.heading_sizes = heading_sizes
        self.boiler = boiler
        self.tables = tables
        self.para: list[str] = []
        self.code: list[str] = []
        self.emitted_tables: set[int] = set()

    def flush_para(self) -> None:
        if self.para:
            text = _HYPHEN_BREAK.sub(r"\1\2", "\n".join(self.para))
            self.asm.add_paragraph(" ".join(text.split("\n")))
            self.para.clear()

    def flush_code(self) -> None:
        if self.code:
            self.asm.add_code("\n".join(self.code))
            self.code.clear()

    def emit(self, lines: list[_Line]) -> None:
        for ln in lines:
            text = ln.text
            if not text or re.sub(r"\d+", "#", text) in self.boiler:
                continue
            if self._in_table(ln):
                continue
            level = PdfParser._heading_level(ln, self.body_size, self.heading_sizes)
            if level:
                self.flush_para()
                self.flush_code()
                self.asm.add_heading(level, text)
            elif ln.mono:
                self.flush_para()
                self.code.append(text)
            else:
                self.flush_code()
                self.para.append(text)
        self.flush_para()
        self.flush_code()

    def _in_table(self, ln: _Line) -> bool:
        for ti, (bbox, md, n_rows, n_cols) in enumerate(self.tables):
            if bbox[1] - 2 <= ln.y <= bbox[3] + 2:
                if ti not in self.emitted_tables:
                    self.flush_para()
                    self.flush_code()
                    self.asm.add_table(md, n_rows=n_rows, n_cols=n_cols)
                    self.emitted_tables.add(ti)
                return True
        return False
