"""Citation engine (PRD §6.7) and the extractive answer mode (ADR-010)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from akl.rag.context_builder import BuiltContext, locator

_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class Citation:
    index: int
    chunk_id: str
    lineage_id: str | None
    document_id: str
    title: str | None
    source_type: str | None
    locator: str
    url: str | None
    snippet: str
    score: float | None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class CitedAnswer:
    answer: str
    citations: list[Citation]
    flags: list[str] = field(default_factory=list)
    uncited_ratio: float = 0.0
    mode: str = "generative"


def _citation_for(block_index: int, ctx: BuiltContext, new_index: int) -> Citation:
    block = ctx.blocks[block_index - 1]
    p = block.candidate.payload
    return Citation(
        index=new_index,
        chunk_id=block.chunk_id,
        lineage_id=str(p.get("lineage_id")) if p.get("lineage_id") else None,
        document_id=str(p.get("document_id") or ""),
        title=p.get("title"),
        source_type=p.get("source_type"),
        locator=locator(p),
        url=str(p.get("source_uri") or p.get("canonical_source_uri") or "") or None,
        snippet=" ".join(str(p.get("text") or "").split())[:200],
        score=block.candidate.rerank_score,
    )


def attach_citations(
    answer: str, ctx: BuiltContext, *, max_uncited_ratio: float = 0.2
) -> CitedAnswer:
    """Validate ``[n]`` markers against context blocks, renumber sequentially, compute uncited ratio."""
    valid = range(1, len(ctx.blocks) + 1)
    mapping: dict[int, int] = {}
    flags: list[str] = []

    def replace(match: re.Match[str]) -> str:
        nums = [int(x) for x in match.group(1).split(",")]
        kept: list[str] = []
        for n in nums:
            if n not in valid:
                flags.append("invalid_marker")
                continue
            if n not in mapping:
                mapping[n] = len(mapping) + 1
            kept.append(f"[{mapping[n]}]")
        return "".join(dict.fromkeys(kept))

    rewritten = _MARKER.sub(replace, answer)
    citations = [
        _citation_for(old, ctx, new) for old, new in sorted(mapping.items(), key=lambda kv: kv[1])
    ]
    sentences = [
        s
        for s in re.split(r"(?<=[.!?])\s+|\n+", rewritten)
        if s.strip() and not s.strip().startswith("```")
    ]
    factual = [s for s in sentences if len(s.split()) >= 4]
    uncited = sum(1 for s in factual if not _MARKER.search(s))
    ratio = uncited / len(factual) if factual else 0.0
    if ratio > max_uncited_ratio:
        flags.append("low_faithfulness")
    if not citations:
        flags.append("no_citations")
    return CitedAnswer(
        answer=rewritten,
        citations=citations,
        flags=list(dict.fromkeys(flags)),
        uncited_ratio=round(ratio, 3),
    )


def extractive_answer(ctx: BuiltContext, *, passages: int = 3) -> CitedAnswer:
    """ADR-010: answer = top passages verbatim, each cited — always has citations."""
    chosen = sorted(ctx.blocks, key=lambda b: -(b.candidate.final_score))[:passages]
    chosen.sort(key=lambda b: b.index)
    parts = [f"{' '.join(b.text.split())} [{b.index}]" for b in chosen]
    answer = "\n\n".join(parts)
    cited = attach_citations(answer, ctx, max_uncited_ratio=1.0)
    cited.mode = "extractive"
    return cited
