"""Chunk quality score (PRD §4.9)."""

from __future__ import annotations

import re
from collections import Counter

_TOKEN = re.compile(r"\w+")


def length_fit(tokens: int, target: int, minimum: int, maximum: int) -> float:
    if tokens <= 0:
        return 0.0
    if tokens <= target:
        return (
            max(0.0, min(1.0, (tokens - minimum) / max(1, target - minimum)))
            if tokens < minimum
            else 1.0
        )
    upper = 1.5 * maximum
    if tokens >= upper:
        return 0.0
    return max(0.0, 1.0 - (tokens - target) / (upper - target))


def alnum_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(ch.isalnum() or ch.isspace() for ch in text) / len(text)


def sentence_completeness(text: str, chunk_type: str) -> float:
    if chunk_type in ("code", "table"):
        return 1.0
    stripped = text.strip()
    if not stripped:
        return 0.0
    starts_ok = stripped[0].isupper() or stripped[0].isdigit() or stripped[0] in "-*`[#"
    ends_ok = stripped[-1] in ".!?:`)\"'"
    return (0.5 if starts_ok else 0.0) + (0.5 if ends_ok else 0.0)


def repetition_ratio(text: str) -> float:
    words = [w.lower() for w in _TOKEN.findall(text)]
    if len(words) < 6:
        return 0.0
    grams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
    counts = Counter(grams)
    duplicated = sum(c - 1 for c in counts.values() if c > 1)
    return duplicated / len(grams)


def chunk_quality(
    text: str,
    *,
    chunk_type: str,
    tokens: int,
    target: int,
    minimum: int,
    maximum: int,
    has_heading: bool,
    boilerplate_ratio: float = 0.0,
) -> tuple[float, tuple[str, ...]]:
    """Return ``(score, flags)`` per PRD §4.9."""
    score = (
        0.30 * length_fit(tokens, target, minimum, maximum)
        + 0.20 * alnum_ratio(text)
        + 0.15 * sentence_completeness(text, chunk_type)
        + 0.15 * (1.0 - max(0.0, min(1.0, boilerplate_ratio)))
        + 0.10 * (1.0 if has_heading else 0.0)
        + 0.10 * (1.0 - repetition_ratio(text))
    )
    score = round(max(0.0, min(1.0, score)), 4)
    flags: list[str] = []
    if tokens < minimum:
        flags.append("short")
    if repetition_ratio(text) > 0.3:
        flags.append("repetitive")
    return score, tuple(flags)
