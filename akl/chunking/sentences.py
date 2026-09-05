"""Sentence and clause splitting with character offsets (PRD §4.11)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pysbd

_CLAUSE_BREAK = re.compile(r"(?<=[;:,\u2014])\s+")
_WS_SPLIT = re.compile(r"\s+")


@dataclass(frozen=True)
class Span:
    """A text span with absolute offsets into the document text."""

    text: str
    start: int
    end: int


class SentenceSplitter:
    def __init__(self, language: str = "en") -> None:
        try:
            self._seg = pysbd.Segmenter(language=language, clean=False)
        except ValueError:  # unsupported language code → English rules
            self._seg = pysbd.Segmenter(language="en", clean=False)

    def split(self, text: str, base_offset: int = 0) -> list[Span]:
        """Sentences with offsets; whitespace between sentences belongs to no span."""
        spans: list[Span] = []
        cursor = 0
        for sent in self._seg.segment(text):
            s = sent.strip()
            if not s:
                continue
            idx = text.find(s, cursor)
            if idx < 0:  # segmenter altered text; fall back to sequential placement
                idx = cursor
            spans.append(Span(s, base_offset + idx, base_offset + idx + len(s)))
            cursor = idx + len(s)
        if not spans and text.strip():
            s = text.strip()
            idx = text.find(s)
            spans.append(Span(s, base_offset + idx, base_offset + idx + len(s)))
        return spans


def clause_split(span: Span, min_chars: int = 40) -> list[Span]:
    """Split one sentence at ; : , — boundaries, keeping pieces ≥ ``min_chars`` where possible."""
    pieces: list[Span] = []
    cursor = 0
    for m in _CLAUSE_BREAK.finditer(span.text):
        piece = span.text[cursor : m.start()]
        if len(piece) >= min_chars:
            pieces.append(Span(piece, span.start + cursor, span.start + m.start()))
            cursor = m.end()
    tail = span.text[cursor:]
    if tail.strip():
        pieces.append(Span(tail, span.start + cursor, span.end))
    return pieces or [span]


def window_split(span: Span, max_words: int) -> list[Span]:
    """Last resort: split on whitespace into windows of ``max_words`` words."""
    words = list(_WS_SPLIT.finditer(span.text))
    if not words:
        return [span]
    starts = [0] + [m.end() for m in words]
    out: list[Span] = []
    i = 0
    while i < len(starts):
        j = min(i + max_words, len(starts))
        s = starts[i]
        e = (
            starts[j - 1] + len(span.text[starts[j - 1] :].split(None, 1)[0])
            if j - 1 < len(starts)
            else len(span.text)
        )
        piece = span.text[s:e].strip()
        if piece:
            out.append(Span(piece, span.start + s, span.start + s + len(piece)))
        i = j
    return out or [span]
