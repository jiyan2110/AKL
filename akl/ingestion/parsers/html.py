"""HTML parser (PRD §3.4.4) built on selectolax.

Pipeline: strip chrome (script/style/nav/footer/aside/forms…) → pick the main
content subtree (``<main>``, ``<article>``, ``[role=main]``, else the densest
text subtree) → walk the DOM into assembler blocks (headings, paragraphs, lists,
tables, ``<pre>`` code, images) → title from ``<title>``/``og:title``/``<h1>``,
canonical URL from ``<link rel=canonical>`` on the same host.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from selectolax.parser import HTMLParser as _Selectolax
from selectolax.parser import Node

from akl.ingestion.models import FetchedObject, QualityReport, UnifiedDocument
from akl.ingestion.parsers.base import BaseParser, DocumentAssembler, ParseError

STRIP_SELECTORS = (
    "script,style,noscript,iframe,svg,canvas,template,nav,footer,header,aside,form,button,"
    "[role=navigation],[role=banner],[role=contentinfo],[aria-hidden=true],"
    ".cookie,.cookies,.cookie-banner,.sidebar,.breadcrumb,.breadcrumbs,.toc,.advert,.ads"
)
MAIN_SELECTORS = (
    "main",
    "article",
    "[role=main]",
    "#content",
    "#main",
    ".content",
    ".post",
    ".article",
)
BLOCK_TAGS = {
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "pre",
    "table",
    "ul",
    "ol",
    "img",
    "figure",
    "blockquote",
    "hr",
    "br",
}
_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _text_len(node: Node) -> int:
    return len(_clean(node.text(separator=" ", strip=True) or ""))


def select_main(root: Node) -> Node:
    """Return the main-content node: semantic containers first, else densest large subtree."""
    for sel in MAIN_SELECTORS:
        found = root.css_first(sel)
        if found is not None and _text_len(found) >= 200:
            return found
    body = root.css_first("body") or root
    best, best_score = body, 0.0
    for node in body.css("div,section,article,td"):
        text_len = _text_len(node)
        if text_len < 200:
            continue
        links = sum(_text_len(a) for a in node.css("a"))
        density = (text_len - links) / max(1, len(node.html or ""))
        score = density * text_len**0.5
        if score > best_score:
            best, best_score = node, score
    return best


class HtmlParser(BaseParser):
    name = "html"
    version = "1.0.0"
    mime_types = ("text/html", "application/xhtml+xml")
    extensions = ("html", "htm", "xhtml")
    source_types = ("html", "github")

    def parse(self, obj: FetchedObject) -> UnifiedDocument:
        flags: list[str] = []
        try:
            raw = obj.data.decode("utf-8")
        except UnicodeDecodeError:
            raw = obj.data.decode("utf-8", errors="replace")
            flags.append("encoding_issues")
        try:
            tree = _Selectolax(raw)
        except Exception as exc:  # pragma: no cover - selectolax is very tolerant
            raise ParseError("html parse failed", details={"error": str(exc)}) from exc
        if tree.body is None and tree.root is None:
            raise ParseError("empty html document")

        title = self._title(tree)
        canonical = self._canonical(tree, obj.item.uri)
        for node in tree.css(STRIP_SELECTORS):
            node.decompose()
        main = select_main(tree.root) if tree.root is not None else None
        if main is None:
            raise ParseError("no html root")

        asm = DocumentAssembler()
        self._walk(main, asm, [])
        meta: dict[str, str] = {"html.title": title or "", "html.canonical": canonical or ""}
        for m in tree.css("meta[property^='og:'],meta[name='description'],meta[name='author']"):
            key = m.attributes.get("property") or m.attributes.get("name") or ""
            content = m.attributes.get("content") or ""
            if key and content:
                meta[f"html.{key}"] = _clean(content)[:500]
        meta.update(obj.source_metadata)
        return asm.build(
            source_uri=canonical or obj.item.uri,
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

    # -- helpers ----------------------------------------------------------------------
    @staticmethod
    def _title(tree: _Selectolax) -> str | None:
        t = tree.css_first("title")
        if t is not None and _clean(t.text()):
            return _clean(t.text())
        og = tree.css_first("meta[property='og:title']")
        if og is not None and og.attributes.get("content"):
            return _clean(og.attributes["content"] or "")
        h1 = tree.css_first("h1")
        return _clean(h1.text()) if h1 is not None and _clean(h1.text()) else None

    @staticmethod
    def _canonical(tree: _Selectolax, source_uri: str) -> str | None:
        link = tree.css_first("link[rel='canonical']")
        href = (link.attributes.get("href") if link is not None else None) or ""
        if not href:
            return None
        src, can = urlparse(source_uri), urlparse(href)
        if can.scheme in ("http", "https") and can.netloc.lower() == src.netloc.lower():
            return href
        return None

    def _walk(self, node: Node, asm: DocumentAssembler, buffer: list[str]) -> None:
        """Depth-first walk emitting blocks; inline text accumulates into ``buffer`` until a block boundary."""
        for child in node.iter(include_text=True):
            tag = child.tag
            if tag == "-text":
                text = child.text(strip=False) or ""
                if text.strip():
                    buffer.append(text)
                continue
            if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                self._flush(asm, buffer)
                asm.add_heading(int(tag[1]), _clean(child.text()))
            elif tag == "pre":
                self._flush(asm, buffer)
                code_el = child.css_first("code")
                lang = None
                classes = ((code_el or child).attributes.get("class") or "").split()
                for c in classes:
                    if c.startswith(("language-", "lang-")):
                        lang = c.split("-", 1)[1].lower()
                asm.add_code(child.text(), language=lang)
            elif tag == "table":
                self._flush(asm, buffer)
                md, n_rows, n_cols = self._table(child)
                if md:
                    cap = child.css_first("caption")
                    asm.add_table(
                        md,
                        n_rows=n_rows,
                        n_cols=n_cols,
                        caption=_clean(cap.text()) if cap else None,
                    )
            elif tag in ("ul", "ol"):
                self._flush(asm, buffer)
                items = [_clean(li.text()) for li in child.css("li")]
                asm.add_list([i for i in items if i], ordered=(tag == "ol"))
            elif tag == "img":
                alt = child.attributes.get("alt")
                asm.add_image(alt=_clean(alt) if alt else None)
            elif tag in ("p", "blockquote", "figure", "li", "dd", "dt", "hr"):
                self._flush(asm, buffer)
                if tag != "hr":
                    self._walk(child, asm, buffer)
                    self._flush(asm, buffer)
            elif tag in ("br",):
                buffer.append("\n")
            elif tag in ("a",):
                text = _clean(child.text())
                href = child.attributes.get("href") or ""
                if text:
                    buffer.append(
                        f"[{text}]({href})"
                        if href.startswith(("http://", "https://", "/"))
                        else text
                    )
            elif (
                tag in ("code", "kbd", "samp")
                and child.parent is not None
                and child.parent.tag != "pre"
            ):
                buffer.append(f"`{_clean(child.text())}`")
            elif tag in (
                "div",
                "section",
                "article",
                "main",
                "span",
                "body",
                "html",
                "td",
                "th",
                "tr",
                "tbody",
                "thead",
                "em",
                "strong",
                "b",
                "i",
                "u",
                "small",
                "sup",
                "sub",
                "mark",
                "label",
                "dl",
                "details",
                "summary",
                "center",
                "font",
            ):
                if tag in ("div", "section", "article", "main", "body", "html", "dl", "details"):
                    self._flush(asm, buffer)
                self._walk(child, asm, buffer)
                if tag in ("div", "section", "article", "main", "body", "html", "dl", "details"):
                    self._flush(asm, buffer)
            else:
                self._walk(child, asm, buffer)

    @staticmethod
    def _flush(asm: DocumentAssembler, buffer: list[str]) -> None:
        if buffer:
            text = _clean("".join(buffer))
            if text:
                asm.add_paragraph(text)
            buffer.clear()

    @staticmethod
    def _table(node: Node) -> tuple[str, int, int]:
        rows: list[list[str]] = []
        for tr in node.css("tr"):
            cells = [_clean(c.text()).replace("|", "\\|") for c in tr.css("th,td")]
            if any(cells):
                rows.append(cells)
        if len(rows) < 1:
            return "", 0, 0
        n_cols = max(len(r) for r in rows)
        rows = [r + [""] * (n_cols - len(r)) for r in rows]
        lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * n_cols]
        lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        return "\n".join(lines), len(rows) - 1, n_cols


def looks_like_html(data: bytes) -> bool:
    head = data[:2048].lstrip().lower()
    return head.startswith(b"<!doctype html") or b"<html" in head or b"<body" in head


def html_meta(obj: FetchedObject) -> dict[str, Any]:  # pragma: no cover - convenience
    return {"size": obj.size_bytes, "mime": obj.mime_type}
