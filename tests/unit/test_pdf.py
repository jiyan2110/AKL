"""Unit tests: PDF parser and connector (Milestone 12) — PDFs generated with PyMuPDF."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from akl.ingestion.connectors.pdf import PdfConnector, PdfConnectorConfig
from akl.ingestion.models import CodeBlock, FetchedObject, HeadingBlock, SourceItem
from akl.ingestion.parsers.pdf import PdfEncryptedError, PdfNoTextError, PdfParser, PdfTooLargeError

pytestmark = pytest.mark.unit


def make_pdf(pages: int = 3, *, with_footer: bool = True, mono: bool = True) -> bytes:
    doc = pymupdf.open()
    for p in range(pages):
        page = doc.new_page()
        if with_footer:
            page.insert_text((72, 40), "ACME Internal Handbook", fontsize=9)
            page.insert_text((72, 800), f"Page {p + 1}", fontsize=9)
        page.insert_text((72, 90), f"Chapter {p + 1} Title", fontsize=20)
        page.insert_text((72, 130), "Section A", fontsize=14)
        y = 160
        for i in range(6):
            page.insert_text(
                (72, y), f"Body line {i} of page {p + 1} explains the sys-", fontsize=10
            )
            y += 14
            page.insert_text((72, y), "tem in detail for operators.", fontsize=10)
            y += 14
        if mono:
            page.insert_text((72, y + 10), "print('hello')", fontsize=10, fontname="cour")
    return doc.tobytes()


def fetched(data: bytes) -> FetchedObject:
    item = SourceItem(
        uri="file:///x/h.pdf", canonical_uri="file:///x/h.pdf", source_type="pdf", filename="h.pdf"
    )
    return FetchedObject.from_bytes(item, data, mime_type="application/pdf")


def test_pdf_headings_boilerplate_code_pages() -> None:
    doc = PdfParser().parse(fetched(make_pdf()))
    headings = [b for b in doc.blocks if isinstance(b, HeadingBlock)]
    assert [h.text for h in headings][:2] == ["Chapter 1 Title", "Section A"]
    assert headings[0].level == 1
    assert headings[1].level == 2
    assert "ACME Internal Handbook" not in doc.text
    assert "Page 2" not in doc.text
    assert any(isinstance(b, CodeBlock) and "print" in b.text for b in doc.blocks)
    assert [p["page"] for p in doc.page_map] == [1, 2, 3]
    assert doc.metadata["pdf.pages"] == "3"
    # hyphenation repaired across line breaks
    assert "sys-tem" not in doc.text
    assert "system in detail" in doc.text
    # offsets map into text
    for b in doc.blocks:
        if isinstance(b, HeadingBlock):
            assert doc.text[b.start_char : b.end_char].endswith(b.text)


def test_pdf_too_many_pages() -> None:
    with pytest.raises(PdfTooLargeError):
        PdfParser(max_pages=2).parse(fetched(make_pdf(pages=3)))


def test_pdf_image_only_raises_no_text() -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
    pix.clear_with(200)
    page.insert_image(pymupdf.Rect(72, 72, 172, 172), pixmap=pix)
    with pytest.raises(PdfNoTextError):
        PdfParser().parse(fetched(doc.tobytes()))


def test_pdf_encrypted_without_empty_password() -> None:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret text " * 20)
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    with pytest.raises(PdfEncryptedError):
        PdfParser().parse(fetched(data))


def test_pdf_connector_discovers_only_pdfs(tmp_path: Path) -> None:
    (tmp_path / "a.pdf").write_bytes(make_pdf(1))
    (tmp_path / "b.PDF").write_bytes(make_pdf(1))
    (tmp_path / "c.md").write_text("# no", encoding="utf-8")
    conn = PdfConnector(PdfConnectorConfig(id="pdf-test", type="pdf", root_path=tmp_path))
    items = [e for e in conn.discover({}) if isinstance(e, SourceItem)]
    assert sorted(i.filename or "" for i in items) == ["a.pdf", "b.PDF"]
    obj = conn.fetch(items[0])
    assert obj.mime_type == "application/pdf"
    assert obj.data.startswith(b"%PDF")
