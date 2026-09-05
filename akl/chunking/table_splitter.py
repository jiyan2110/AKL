"""Table-aware splitting (PRD §4.6): header repeated on every row split; wide tables transposed."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

WIDE_TABLE_COLS = 30


@dataclass(frozen=True)
class TablePiece:
    markdown: str
    index: int
    total: int
    row_start: int  # 0-based body row index of first row in this piece
    row_end: int  # exclusive


def parse_markdown_table(md: str) -> tuple[list[str], list[list[str]]]:
    lines = [ln.strip() for ln in md.strip().splitlines() if ln.strip().startswith("|")]
    if not lines:
        return [], []

    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    header = cells(lines[0])
    body = [cells(ln) for ln in lines[1:] if not set(ln.replace("|", "").strip()) <= set("-: ")]
    return header, body


def render_markdown_table(header: list[str], rows: list[list[str]]) -> str:
    n = len(header)
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * n]
    out += ["| " + " | ".join((r + [""] * n)[:n]) + " |" for r in rows]
    return "\n".join(out)


def transpose_wide_table(header: list[str], rows: list[list[str]]) -> str:
    """Row-wise ``key: value`` records for tables with very many columns."""
    records = []
    for i, r in enumerate(rows, start=1):
        pairs = [f"{h}: {v}" for h, v in zip(header, r, strict=False) if v]
        records.append(f"Row {i}: " + "; ".join(pairs))
    return "\n".join(records)


def split_table(
    md: str, count: Callable[[str], int], max_tokens: int, caption: str | None = None
) -> list[TablePiece]:
    header, body = parse_markdown_table(md)
    if not header:
        return [TablePiece(md, 0, 1, 0, 0)]
    if len(header) >= WIDE_TABLE_COLS:
        text = transpose_wide_table(header, body)
        return [TablePiece(text, 0, 1, 0, len(body))]
    prefix = f"{caption}\n\n" if caption else ""
    full = prefix + render_markdown_table(header, body)
    if count(full) <= max_tokens:
        return [TablePiece(full, 0, 1, 0, len(body))]

    pieces: list[tuple[int, int]] = []
    start = 0
    while start < len(body):
        end = start + 1
        while (
            end < len(body)
            and count(prefix + render_markdown_table(header, body[start : end + 1])) <= max_tokens
        ):
            end += 1
        pieces.append((start, end))
        start = end
    total = len(pieces)
    return [
        TablePiece(prefix + render_markdown_table(header, body[s:e]), i, total, s, e)
        for i, (s, e) in enumerate(pieces)
    ]
