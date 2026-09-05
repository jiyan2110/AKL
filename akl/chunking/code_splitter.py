"""Code-aware splitting (PRD §4.5): definition boundaries → blank-line groups → line windows.

No tree-sitter dependency: top-level definition starts are detected with per-language
regexes (functions, classes, methods, impl blocks). Splits are contiguous slices of the
code block, so offsets stay exact; the block's first line is prepended as context to
every non-first piece via ``context_lines`` (carried in the chunk's context prefix).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

_DEF_PATTERNS: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"^(?:async\s+def|def|class)\s+\w+", re.MULTILINE),
    "javascript": re.compile(
        r"^(?:export\s+)?(?:async\s+)?(?:function\s+\w+|class\s+\w+|const\s+\w+\s*=\s*(?:async\s*)?\()",
        re.MULTILINE,
    ),
    "typescript": re.compile(
        r"^(?:export\s+)?(?:async\s+)?(?:function\s+\w+|class\s+\w+|interface\s+\w+|type\s+\w+|const\s+\w+\s*=\s*(?:async\s*)?\()",
        re.MULTILINE,
    ),
    "go": re.compile(r"^(?:func\s+(?:\([^)]*\)\s*)?\w+|type\s+\w+)", re.MULTILINE),
    "java": re.compile(
        r"^\s{0,4}(?:public|private|protected|static|final|abstract|\s)*\s*(?:class|interface|enum|[\w<>\[\]]+\s+\w+\s*\()",
        re.MULTILINE,
    ),
    "rust": re.compile(
        r"^(?:pub(?:\([^)]*\))?\s+)?(?:fn|struct|enum|impl|trait|mod)\s+\w+", re.MULTILINE
    ),
    "sql": re.compile(r"^(?:CREATE|ALTER|INSERT|WITH|SELECT|DROP)\b", re.IGNORECASE | re.MULTILINE),
    "bash": re.compile(r"^(?:function\s+\w+|\w+\s*\(\)\s*\{)", re.MULTILINE),
}
_DEF_PATTERNS["ts"] = _DEF_PATTERNS["typescript"]
_DEF_PATTERNS["js"] = _DEF_PATTERNS["javascript"]
_DEF_PATTERNS["py"] = _DEF_PATTERNS["python"]
_DEF_PATTERNS["sh"] = _DEF_PATTERNS["bash"]


@dataclass(frozen=True)
class CodePiece:
    text: str
    start: int  # offset within the code block text
    end: int
    index: int
    total: int


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n" and i + 1 < len(text):
            starts.append(i + 1)
    return starts


def _slices_from_boundaries(text: str, boundaries: list[int]) -> list[tuple[int, int]]:
    bounds = sorted(set([0, *boundaries, len(text)]))
    return [
        (bounds[i], bounds[i + 1])
        for i in range(len(bounds) - 1)
        if text[bounds[i] : bounds[i + 1]].strip()
    ]


def _pack(
    text: str, slices: list[tuple[int, int]], count: Callable[[str], int], max_tokens: int
) -> list[tuple[int, int]]:
    """Greedily merge adjacent slices while the packed piece stays ≤ max_tokens."""
    out: list[tuple[int, int]] = []
    for s, e in slices:
        if out and count(text[out[-1][0] : e]) <= max_tokens:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def split_code(
    text: str, language: str | None, count: Callable[[str], int], max_tokens: int
) -> list[CodePiece]:
    """Split ``text`` into pieces ≤ ``max_tokens`` each (best effort; a single giant line is kept whole)."""
    if count(text) <= max_tokens:
        return [CodePiece(text, 0, len(text), 0, 1)]

    pattern = _DEF_PATTERNS.get((language or "").lower())
    slices: list[tuple[int, int]] | None = None
    if pattern is not None:
        boundaries = [m.start() for m in pattern.finditer(text) if m.start() > 0]
        if boundaries:
            slices = _pack(text, _slices_from_boundaries(text, boundaries), count, max_tokens)
    if slices is None or any(count(text[s:e]) > max_tokens for s, e in slices):
        # blank-line groups
        boundaries = [m.end() for m in re.finditer(r"\n[ \t]*\n", text)]
        slices = (
            _pack(text, _slices_from_boundaries(text, boundaries), count, max_tokens)
            if boundaries
            else None
        )
    if slices is None or any(count(text[s:e]) > max_tokens for s, e in slices):
        # line windows
        starts = _line_starts(text)
        pieces: list[tuple[int, int]] = []
        begin = 0
        for i in range(1, len(starts)):
            if count(text[begin : starts[i]]) > max_tokens and starts[i - 1] > begin:
                pieces.append((begin, starts[i - 1]))
                begin = starts[i - 1]
        pieces.append((begin, len(text)))
        slices = pieces
    total = len(slices)
    return [
        CodePiece(text[s:e].rstrip("\n"), s, s + len(text[s:e].rstrip("\n")), i, total)
        for i, (s, e) in enumerate(slices)
    ]
