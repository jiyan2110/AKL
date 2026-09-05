"""Document quality score (PRD §3.11)."""

from __future__ import annotations

from akl.ingestion.models import HeadingBlock, UnifiedDocument
from akl.ingestion.validators import text_density


def document_quality(
    doc: UnifiedDocument, *, language_confidence: float, boilerplate_ratio: float = 0.0
) -> float:
    """``0.35·density + 0.20·structure + 0.15·lang_conf + 0.15·(1−boilerplate) + 0.15·length``."""
    words = len(doc.text.split())
    headings = sum(1 for b in doc.blocks if isinstance(b, HeadingBlock))
    structure = min(1.0, headings / max(1.0, words / 400))
    length = min(1.0, words / 300)
    score = (
        0.35 * text_density(doc.text)
        + 0.20 * structure
        + 0.15 * max(0.0, min(1.0, language_confidence))
        + 0.15 * (1.0 - max(0.0, min(1.0, boilerplate_ratio)))
        + 0.15 * length
    )
    return round(max(0.0, min(1.0, score)), 4)
