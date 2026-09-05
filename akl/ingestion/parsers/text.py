"""Plain-text, reStructuredText-lite and code parsers for repository files (PRD §3.4.5).

* :class:`TextParser` — ``.txt``/``.rst``: paragraphs split on blank lines; RST
  underlined titles (``=====``/``-----``) and ATX-style headings become headings.
* :class:`CodeParser` — source files become a single code block tagged with the
  language; the Code Chunker (Milestone 17) splits it at definition boundaries.
"""

from __future__ import annotations

import re

from akl.ingestion.models import FetchedObject, QualityReport, UnifiedDocument
from akl.ingestion.parsers.base import BaseParser, DocumentAssembler

_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+]){3,}\s*$")
_ATX = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_LEVEL_BY_CHAR = {"=": 1, "-": 2, "~": 3, "^": 4, '"': 5, "'": 6, "`": 6, "#": 1, "*": 2, "+": 3}

CODE_LANGUAGES: dict[str, str] = {
    "py": "python",
    "ts": "typescript",
    "tsx": "tsx",
    "js": "javascript",
    "jsx": "jsx",
    "go": "go",
    "java": "java",
    "rs": "rust",
    "sql": "sql",
    "yaml": "yaml",
    "yml": "yaml",
    "json": "json",
    "sh": "bash",
    "toml": "toml",
    "c": "c",
    "h": "c",
    "cpp": "cpp",
    "cs": "csharp",
    "rb": "ruby",
    "kt": "kotlin",
    "tf": "hcl",
}


def _decode(obj: FetchedObject) -> tuple[str, tuple[str, ...]]:
    try:
        return obj.data.decode("utf-8"), ()
    except UnicodeDecodeError:
        return obj.data.decode("utf-8", errors="replace"), ("encoding_issues",)


class TextParser(BaseParser):
    name = "text"
    version = "1.0.0"
    mime_types = ("text/plain", "text/x-rst")
    extensions = ("txt", "rst", "text")
    source_types = ("github", "markdown", "html")

    def parse(self, obj: FetchedObject) -> UnifiedDocument:
        raw, flags = _decode(obj)
        lines = raw.replace("\r\n", "\n").split("\n")
        asm = DocumentAssembler()
        para: list[str] = []
        code: list[str] = []
        in_code = False

        def flush() -> None:
            if para:
                asm.add_paragraph(" ".join(s.strip() for s in para))
                para.clear()

        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip().startswith("```"):
                flush()
                if in_code:
                    asm.add_code("\n".join(code))
                    code.clear()
                in_code = not in_code
                i += 1
                continue
            if in_code:
                code.append(line)
                i += 1
                continue
            m = _ATX.match(line)
            if m:
                flush()
                asm.add_heading(len(m.group(1)), m.group(2))
                i += 1
                continue
            if (
                i + 1 < len(lines)
                and line.strip()
                and _UNDERLINE.match(lines[i + 1])
                and len(lines[i + 1].strip()) >= len(line.strip())
            ):
                flush()
                asm.add_heading(_LEVEL_BY_CHAR.get(lines[i + 1].strip()[0], 2), line.strip())
                i += 2
                continue
            if not line.strip():
                flush()
            else:
                para.append(line)
            i += 1
        flush()
        if code:
            asm.add_code("\n".join(code))
        return asm.build(
            source_uri=obj.item.uri,
            source_type=obj.item.source_type,
            content_sha256=obj.sha256,
            parser_name=self.name,
            parser_version=self.version,
            security_level=obj.item.security_level,
            allowed_groups=obj.item.allowed_groups,
            metadata=dict(obj.source_metadata),
            quality=QualityReport(score=1.0, flags=flags),
        )


class CodeParser(BaseParser):
    name = "code"
    version = "1.0.0"
    mime_types = ()
    extensions = tuple(CODE_LANGUAGES)
    source_types = ("github",)

    def parse(self, obj: FetchedObject) -> UnifiedDocument:
        raw, flags = _decode(obj)
        language = CODE_LANGUAGES.get(obj.extension, obj.extension or None)
        asm = DocumentAssembler()
        asm.add_heading(1, obj.item.source_metadata.get("git.path") or obj.item.filename or "code")
        asm.add_code(raw, language=language)
        return asm.build(
            source_uri=obj.item.uri,
            source_type=obj.item.source_type,
            content_sha256=obj.sha256,
            parser_name=self.name,
            parser_version=self.version,
            security_level=obj.item.security_level,
            allowed_groups=obj.item.allowed_groups,
            metadata={**obj.source_metadata, "code.language": language or ""},
            quality=QualityReport(score=1.0, flags=flags),
        )
