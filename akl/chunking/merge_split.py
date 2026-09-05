"""Token pass: split oversized candidates, merge undersized ones, sentence-aligned overlap (PRD §4.7, §4.11)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from akl.chunking.sentences import Span, clause_split, window_split

Counter = Callable[[str], int]


@dataclass
class Candidate:
    """A run of sentence spans destined to become one prose chunk."""

    spans: list[Span] = field(default_factory=list)
    overlap: list[Span] = field(
        default_factory=list
    )  # leading spans borrowed from the previous candidate

    @property
    def start(self) -> int:
        return (self.overlap or self.spans)[0].start

    @property
    def end(self) -> int:
        return self.spans[-1].end

    def text(self, doc_text: str) -> str:
        return doc_text[self.start : self.end]


def atomize(span: Span, count: Counter, max_tokens: int) -> list[Span]:
    """Break a single sentence that exceeds ``max_tokens`` into clause/window pieces."""
    if count(span.text) <= max_tokens:
        return [span]
    pieces = clause_split(span)
    if len(pieces) == 1 or any(count(p.text) > max_tokens for p in pieces):
        words_per = max(8, int(max_tokens * 0.6))
        pieces = [w for p in pieces for w in window_split(p, words_per)]
    return pieces


def pack(
    spans: Sequence[Span],
    count: Counter,
    target: int,
    max_tokens: int,
    boundaries: set[int] | None = None,
) -> list[Candidate]:
    """Greedy accumulation up to ``target`` tokens; forced cuts at ``boundaries`` (sentence indices)."""
    boundaries = boundaries or set()
    out: list[Candidate] = []
    cur = Candidate()
    cur_tokens = 0
    for i, span in enumerate(spans):
        t = count(span.text)
        if cur.spans and (i in boundaries or cur_tokens + t > target):
            out.append(cur)
            cur, cur_tokens = Candidate(), 0
        cur.spans.append(span)
        cur_tokens += t
    if cur.spans:
        out.append(cur)
    # hard limit: any candidate over max is split at sentence boundaries
    fixed: list[Candidate] = []
    for cand in out:
        if count(" ".join(s.text for s in cand.spans)) <= max_tokens:
            fixed.append(cand)
            continue
        part = Candidate()
        part_tokens = 0
        for span in cand.spans:
            t = count(span.text)
            if part.spans and part_tokens + t > max_tokens:
                fixed.append(part)
                part, part_tokens = Candidate(), 0
            part.spans.append(span)
            part_tokens += t
        if part.spans:
            fixed.append(part)
    return fixed


def merge_undersized(
    cands: list[Candidate], count: Counter, min_tokens: int, max_tokens: int
) -> list[Candidate]:
    """Merge candidates below ``min_tokens`` into a neighbour when the result fits ``max_tokens``."""
    i = 0
    while i < len(cands):
        c = cands[i]
        tokens = count(" ".join(s.text for s in c.spans))
        if tokens >= min_tokens:
            i += 1
            continue
        if (
            i + 1 < len(cands)
            and tokens + count(" ".join(s.text for s in cands[i + 1].spans)) <= max_tokens
        ):
            cands[i + 1].spans = c.spans + cands[i + 1].spans
            del cands[i]
            continue
        if i > 0 and tokens + count(" ".join(s.text for s in cands[i - 1].spans)) <= max_tokens:
            cands[i - 1].spans = cands[i - 1].spans + c.spans
            del cands[i]
            continue
        i += 1
    return cands


def apply_overlap(cands: list[Candidate], count: Counter, overlap_tokens: int) -> list[Candidate]:
    """Prepend the trailing sentences of the previous candidate (≤ overlap_tokens, sentence-aligned)."""
    if overlap_tokens <= 0:
        return cands
    for prev, cur in zip(cands, cands[1:], strict=False):
        tail: list[Span] = []
        used = 0
        for span in reversed(prev.spans):
            t = count(span.text)
            if used + t > overlap_tokens:
                break
            tail.insert(0, span)
            used += t
        cur.overlap = tail
    return cands
