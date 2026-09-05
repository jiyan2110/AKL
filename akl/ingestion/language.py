"""Language detection (PRD §3.4.6) via langdetect (deterministic seed)."""

from __future__ import annotations

from langdetect import DetectorFactory, LangDetectException, detect_langs

from akl.ingestion.models import Block, CodeBlock, TableBlock

DetectorFactory.seed = 0
CONFIDENCE_MIN = 0.7
SAMPLE_CHARS = 5000


def prose_sample(text: str, blocks: tuple[Block, ...]) -> str:
    """First ``SAMPLE_CHARS`` of prose (code/table blocks excluded)."""
    parts: list[str] = []
    total = 0
    for b in blocks:
        if isinstance(b, CodeBlock | TableBlock):
            continue
        segment = text[b.start_char : b.end_char]
        parts.append(segment)
        total += len(segment)
        if total >= SAMPLE_CHARS:
            break
    return "\n".join(parts)[:SAMPLE_CHARS]


def detect_language(sample: str) -> tuple[str, float]:
    """Return ``(iso639-1, confidence)``; ``("und", 0.0)`` when undetermined."""
    if len(sample.strip()) < 20:
        return "und", 0.0
    try:
        candidates = detect_langs(sample)
    except LangDetectException:
        return "und", 0.0
    if not candidates:
        return "und", 0.0
    best = candidates[0]
    lang = str(best.lang).split("-")[0]
    prob = float(best.prob)
    return (lang, prob) if prob >= CONFIDENCE_MIN else ("und", prob)
