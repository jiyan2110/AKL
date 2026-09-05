"""Sparse-retrieval tokenizer (PRD §6.3.2).

lowercase → Unicode word tokens → identifiers split on ``_ . / -`` and camelCase humps
(emitting both the whole identifier and its parts) → light Snowball-like suffix stemming for
plain prose words only (identifiers are never stemmed) → stopwords dropped unless the query
is very short.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-/]*[A-Za-z0-9]|[A-Za-z0-9]", re.UNICODE)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_IDENT_SPLIT = re.compile(r"[_./\-]+")
STOPWORDS = frozenset(
    "a an the and or but if then else of to in on at by for with from as is are was were be been being it its this that these those there here "
    "do does did doing have has had having i you he she we they me him her us them my your our their what which who whom how when where why "
    "not no yes can could should would will shall may might must about into over under again further once all any both each few more most "
    "other some such than too very s t just also".split()
)
_SUFFIXES = (
    "ization",
    "ations",
    "ation",
    "ingly",
    "ness",
    "ments",
    "ment",
    "ings",
    "ing",
    "edly",
    "ies",
    "ers",
    "ed",
    "es",
    "er",
    "ly",
    "s",
)


def _stem(word: str) -> str:
    if len(word) <= 4:
        return word
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            stem = word[: -len(suf)]

            if suf == "es" and not stem.endswith(("s", "x", "z", "ch", "sh")):
                continue
    return word


def tokenize(text: str, *, keep_stopwords: bool | None = None) -> list[str]:
    raw = _WORD.findall(text)
    keep = keep_stopwords if keep_stopwords is not None else len(raw) <= 3
    out: list[str] = []
    for tok in raw:
        is_identifier = bool(_IDENT_SPLIT.search(tok)) or bool(_CAMEL.search(tok))
        low = tok.lower()
        if is_identifier:
            out.append(low)  # whole identifier, exact
            parts = [p for piece in _IDENT_SPLIT.split(tok) for p in _CAMEL.split(piece) if p]
            out.extend(p.lower() for p in parts if len(p) > 1)
            continue
        if not keep and low in STOPWORDS:
            continue
        out.append(_stem(low))
    return out
