"""Markdown parser: markdown-it AST → UnifiedDocument (PRD §3.4.3)."""

from __future__ import annotations

import re
from typing import Any

import yaml
from markdown_it import MarkdownIt
from markdown_it.token import Token
from mdit_py_plugins.front_matter import front_matter_plugin

from akl.ingestion.models import FetchedObject, QualityReport, SecurityLevel, UnifiedDocument
from akl.ingestion.parsers.base import BaseParser, DocumentAssembler, ParseError

_MDX_IMPORT_EXPORT = re.compile(r"^(?:import|export)\s.*?$\n?", re.MULTILINE)
_JSX_TAG = re.compile(r"</?[A-Z][A-Za-z0-9.]*(?:\s[^<>]*)?/?>")
_HTML_TAG = re.compile(r"<[^>]+>")
_LEVELS: dict[str, int] = {"public": 0, "internal": 1, "restricted": 2}


def strip_mdx(text: str) -> str:
    return _JSX_TAG.sub("", _MDX_IMPORT_EXPORT.sub("", text))


def render_inline(token: Token | None) -> str:
    """Flatten an ``inline`` token's children into lightweight Markdown text."""
    if token is None:
        return ""
    if not token.children:
        return token.content.strip()
    out: list[str] = []
    hrefs: list[str] = []
    for child in token.children:
        t = child.type
        if t == "text":
            out.append(child.content)
        elif t == "code_inline":
            out.append(f"`{child.content}`")
        elif t == "softbreak":
            out.append(" ")
        elif t == "hardbreak":
            out.append("\n")
        elif t == "link_open":
            hrefs.append(str(child.attrGet("href") or ""))
            out.append("[")
        elif t == "link_close":
            out.append(f"]({hrefs.pop() if hrefs else ''})")
        elif t == "image":
            out.append(f"![{child.content or 'image'}]")
        elif t == "html_inline":
            out.append(_HTML_TAG.sub("", child.content))
        # strong/em/s open+close: keep text, drop markers
    return "".join(out).strip()


def parse_frontmatter(raw: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _find_close(tokens: list[Token], start: int) -> int:
    """Index just past the token closing the container opened at ``start`` (same nesting level)."""
    level = tokens[start].level
    for j in range(start + 1, len(tokens)):
        if tokens[j].nesting == -1 and tokens[j].level == level:
            return j + 1
    return len(tokens)


def table_markdown(tokens: list[Token]) -> tuple[str, int, int]:
    rows: list[list[str]] = []
    row: list[str] | None = None
    for tok in tokens:
        if tok.type == "tr_open":
            row = []
        elif tok.type == "inline" and row is not None:
            row.append(render_inline(tok).replace("|", "\\|"))
        elif tok.type == "tr_close" and row is not None:
            rows.append(row)
            row = None
    if not rows:
        return "", 0, 0
    n_cols = max(len(r) for r in rows)
    rows = [r + [""] * (n_cols - len(r)) for r in rows]
    header, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * n_cols]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines), len(body), n_cols


def _to_str(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def raise_only(current: SecurityLevel, requested: Any) -> SecurityLevel:
    """Source metadata may raise the security level but never lower it (PRD §9.3)."""
    if (
        isinstance(requested, str)
        and requested in _LEVELS
        and _LEVELS[requested] > _LEVELS[current]
    ):
        return requested  # type: ignore[return-value]
    return current


class MarkdownParser(BaseParser):
    name = "markdown"
    version = "1.0.0"
    mime_types = ("text/markdown", "text/x-markdown")
    extensions = ("md", "mdx", "markdown")
    source_types = ("markdown", "github")

    def __init__(self) -> None:
        self._md = (
            MarkdownIt("commonmark", {"html": True})
            .enable(["table", "strikethrough"])
            .use(front_matter_plugin)
        )

    def parse(self, obj: FetchedObject) -> UnifiedDocument:
        flags: list[str] = []
        try:
            raw = obj.data.decode("utf-8")
        except UnicodeDecodeError:
            raw = obj.data.decode("utf-8", errors="replace")
            flags.append("encoding_issues")
        if obj.extension == "mdx":
            raw = strip_mdx(raw)
        try:
            tokens = self._md.parse(raw)
        except Exception as exc:  # markdown-it is permissive; guard anyway
            raise ParseError("markdown tokenisation failed", details={"error": str(exc)}) from exc

        asm = DocumentAssembler()
        frontmatter: dict[str, Any] = {}
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            t = tok.type
            if t == "front_matter":
                frontmatter = parse_frontmatter(tok.content)
            elif t == "heading_open":
                asm.add_heading(int(tok.tag[1]), render_inline(tokens[i + 1]))
                i += 2
            elif t == "paragraph_open":
                asm.add_paragraph(render_inline(tokens[i + 1]))
                i += 2
            elif t in ("fence", "code_block"):
                info = (tok.info or "").strip()
                asm.add_code(tok.content, language=info.split()[0].lower() if info else None)
            elif t in ("bullet_list_open", "ordered_list_open"):
                end = _find_close(tokens, i)
                items = [render_inline(x) for x in tokens[i:end] if x.type == "inline"]
                asm.add_list(items, ordered=(t == "ordered_list_open"))
                i = end
                continue
            elif t == "table_open":
                end = _find_close(tokens, i)
                md, n_rows, n_cols = table_markdown(tokens[i:end])
                asm.add_table(md, n_rows=n_rows, n_cols=n_cols)
                i = end
                continue
            elif t == "html_block":
                text = _HTML_TAG.sub("", tok.content).strip()
                if text:
                    asm.add_paragraph(text)
            i += 1

        metadata = {f"frontmatter.{k}": _to_str(v) for k, v in frontmatter.items()}
        metadata.update(obj.source_metadata)
        title = _to_str(frontmatter["title"]) if frontmatter.get("title") else None
        level = raise_only(obj.item.security_level, frontmatter.get("security_level"))
        groups = tuple(obj.item.allowed_groups)
        fm_groups = frontmatter.get("allowed_groups")
        if isinstance(fm_groups, list) and fm_groups:
            groups = tuple(str(g) for g in fm_groups)

        return asm.build(
            source_uri=obj.item.uri,
            source_type=obj.item.source_type,
            content_sha256=obj.sha256,
            parser_name=self.name,
            parser_version=self.version,
            title=title,
            language=None,  # language detection runs in the validation pipeline
            security_level=level,
            allowed_groups=groups,
            metadata=metadata,
            quality=QualityReport(score=1.0, flags=tuple(flags)),
        )
