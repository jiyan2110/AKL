"""Context builder (PRD §6.4): dedupe → neighbour expansion → order → token budget → render blocks."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from akl.rag.retrieval.models import Candidate

_TOKEN = re.compile(r"\w+")


@dataclass
class ContextBlock:
    index: int  # 1-based [n] marker
    candidate: Candidate
    text: str
    tokens: int
    truncated: bool = False

    @property
    def chunk_id(self) -> str:
        return self.candidate.chunk_id


@dataclass
class BuiltContext:
    blocks: list[ContextBlock]
    total_tokens: int
    dropped: list[str] = field(default_factory=list)  # chunk_ids removed by dedupe/budget
    flags: list[str] = field(default_factory=list)

    def render(self) -> str:
        parts = []
        for b in self.blocks:
            p = b.candidate.payload
            header = f'[{b.index}] source={p.get("source_type")} title="{p.get("title") or ""}" locator="{locator(p)}" chunk_id={b.chunk_id}'
            prefix = p.get("context_prefix") or ""
            parts.append("\n".join(x for x in (header, prefix, b.text) if x))
        return "\n\n".join(parts)


def trigrams(text: str) -> set[str]:
    words = [w.lower() for w in _TOKEN.findall(text)]
    return {" ".join(words[i : i + 3]) for i in range(max(0, len(words) - 2))} or set(words)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def locator(p: dict[str, Any]) -> str:
    """Human-readable citation locator per source type (PRD §4.14)."""
    src = p.get("source_type")
    if src == "pdf":
        ps, pe = p.get("page_start"), p.get("page_end")
        pages = f"p. {ps}" + (f"–{pe}" if pe and pe != ps else "") if ps else ""
        return f"{p.get('title') or p.get('canonical_source_uri')}, {pages}".rstrip(", ")
    if src in ("markdown", "github"):
        path = p.get("path") or str(p.get("canonical_source_uri") or "").rsplit("/", 1)[-1]
        ls, le = p.get("line_start"), p.get("line_end")
        lines = f"#L{ls}-L{le}" if ls and le else ""
        branch = f" @ {p['branch']}" if p.get("branch") else ""
        repo = f"{p['repo']}/" if p.get("repo") else ""
        return f"{repo}{path}{lines}{branch}"
    if src == "html":
        url = str(p.get("canonical_source_uri") or "")
        words = " ".join(str(p.get("text") or "").split()[:8])
        return f"{url}#:~:text={words}" if words else url
    return str(p.get("canonical_source_uri") or p.get("chunk_id"))


def _trim_overlap(text: str, overlap_tokens: int) -> str:
    """Drop the leading overlap sentences when the previous chunk is already in the context."""
    if overlap_tokens <= 0:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) < 2:
        return text
    # heuristically drop as many leading sentences as the overlap token count covers
    budget = overlap_tokens
    dropped = 0
    for s in sentences[:-1]:
        n = len(_TOKEN.findall(s))
        if n > budget:
            break
        budget -= n
        dropped += 1
    return " ".join(sentences[dropped:]) if dropped else text


class ContextBuilder:
    def __init__(
        self,
        count_tokens: Callable[[str], int],
        *,
        budget_tokens: int,
        top_k: int,
        dedupe_jaccard: float = 0.85,
    ) -> None:
        self.count = count_tokens
        self.budget = budget_tokens
        self.top_k = top_k
        self.dedupe_jaccard = dedupe_jaccard

    def build(self, candidates: Sequence[Candidate]) -> BuiltContext:
        flags: list[str] = []
        picked: list[Candidate] = []
        dropped: list[str] = []
        grams: list[set[str]] = []
        for cand in candidates:
            if len(picked) >= self.top_k:
                break
            g = trigrams(cand.text)
            if any(jaccard(g, other) >= self.dedupe_jaccard for other in grams):
                dropped.append(cand.chunk_id)
                continue
            picked.append(cand)
            grams.append(g)
        if dropped:
            flags.append("deduplicated")

        # order: documents by best score, chunks within a document by chunk_index (PRD §6.4 step 4)
        best_by_doc: dict[str, float] = {}
        for c in picked:
            best_by_doc[c.document_id] = max(best_by_doc.get(c.document_id, -1.0), c.final_score)
        picked.sort(
            key=lambda c: (
                -best_by_doc[c.document_id],
                c.document_id,
                int(c.payload.get("chunk_index") or 0),
            )
        )

        picked_ids = {c.chunk_id for c in picked}
        blocks: list[ContextBlock] = []
        total = 0
        for cand in picked:
            text = cand.text
            prev_id = cand.payload.get("prev_chunk_id")
            if prev_id and str(prev_id) in picked_ids:
                text = _trim_overlap(text, int(cand.payload.get("overlap_prev_tokens") or 0))
            tokens = self.count(text)
            if total + tokens > self.budget:
                remaining = self.budget - total
                if remaining < 32 or blocks:
                    dropped.append(cand.chunk_id)
                    flags.append("budget_truncated")
                    break
                text = _cut_to_budget(text, remaining, self.count)
                tokens = self.count(text)
                flags.append("budget_truncated")
                blocks.append(ContextBlock(len(blocks) + 1, cand, text, tokens, truncated=True))
                total += tokens
                break
            blocks.append(ContextBlock(len(blocks) + 1, cand, text, tokens))
            total += tokens
        return BuiltContext(
            blocks=blocks, total_tokens=total, dropped=dropped, flags=list(dict.fromkeys(flags))
        )


def _cut_to_budget(text: str, budget: int, count: Callable[[str], int]) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    for s in sentences:
        if count(" ".join([*out, s])) > budget:
            break
        out.append(s)
    return " ".join(out) if out else text[: budget * 4]
