"""Rule-based intent classification (PRD §6.2.3)."""

from __future__ import annotations

import re
from enum import StrEnum


class Intent(StrEnum):
    FACTUAL_LOOKUP = "factual_lookup"
    HOW_TO = "how_to"
    CODE_SEARCH = "code_search"
    TROUBLESHOOTING = "troubleshooting"
    COMPARISON = "comparison"
    SUMMARIZATION = "summarization"
    CHITCHAT = "chitchat"


_RULES: list[tuple[Intent, re.Pattern[str]]] = [
    (
        Intent.CHITCHAT,
        re.compile(
            r"^(hi|hello|hey|thanks|thank you|ok|okay|bye|good (morning|evening))[.!]*$", re.I
        ),
    ),
    (
        Intent.TROUBLESHOOTING,
        re.compile(
            r"\b(error|fails?|failing|failed|exception|traceback|crash|broken|not working|timeout|AKL-E\d{4})\b",
            re.I,
        ),
    ),
    (
        Intent.COMPARISON,
        re.compile(
            r"\b(difference between|differences between|vs\.?|versus|compare|compared to|better than)\b",
            re.I,
        ),
    ),
    (
        Intent.SUMMARIZATION,
        re.compile(
            r"\b(summari[sz]e|summary of|overview of|tl;?dr|explain (the )?(chapter|section|document))\b",
            re.I,
        ),
    ),
    (
        Intent.CODE_SEARCH,
        re.compile(
            r"\b(function|class|method|module|snippet|implementation|code|script|def |regex)\b|`[^`]+`",
            re.I,
        ),
    ),
    (
        Intent.HOW_TO,
        re.compile(
            r"^(how (do|can|should|to)|steps? to|configure|set ?up|install|enable|disable)\b|\bhow to\b",
            re.I,
        ),
    ),
]

# retrieval knobs per intent (PRD §6.2.3 "Effect" column)
INTENT_PROFILES: dict[Intent, dict[str, object]] = {
    Intent.FACTUAL_LOOKUP: {"top_k": 8, "sparse_weight": 1.0, "boost_chunk_types": ()},
    Intent.HOW_TO: {
        "top_k": 10,
        "sparse_weight": 1.0,
        "boost_chunk_types": ("prose", "code", "list"),
    },
    Intent.CODE_SEARCH: {"top_k": 8, "sparse_weight": 1.3, "boost_chunk_types": ("code",)},
    Intent.TROUBLESHOOTING: {
        "top_k": 8,
        "sparse_weight": 1.3,
        "boost_chunk_types": ("prose", "list"),
    },
    Intent.COMPARISON: {"top_k": 10, "sparse_weight": 1.0, "boost_chunk_types": ()},
    Intent.SUMMARIZATION: {"top_k": 12, "sparse_weight": 0.8, "boost_chunk_types": ()},
    Intent.CHITCHAT: {"top_k": 0, "sparse_weight": 0.0, "boost_chunk_types": ()},
}


def classify_intent(text: str) -> Intent:
    for intent, pattern in _RULES:
        if pattern.search(text):
            return intent
    return Intent.FACTUAL_LOOKUP
