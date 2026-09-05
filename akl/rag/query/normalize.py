"""Query normalisation with protected code-like tokens (PRD §6.2.1)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from akl.errors import AKLError

# identifiers, paths, error codes, env vars, versions, URLs — never spell-corrected or lower-cased away
PROTECTED = re.compile(
    r"(?:https?://\S+)|(?:[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+)|(?:AKL-[EW]\d{4})|(?:\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b)"
    r"|(?:\b[a-z]+(?:_[a-z0-9]+)+\b)|(?:\b[a-z]+(?:[A-Z][a-z0-9]+)+\b)|(?:\bv?\d+\.\d+(?:\.\d+)?\b)|(?:`[^`]+`)"
)
_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TOKEN = re.compile(r"[\w'`./:-]+", re.UNICODE)


class EmptyQueryError(AKLError):
    code = "AKL-E6001"
    http_status = 422
    retryable = False


@dataclass(frozen=True)
class NormalizedQuery:
    original: str
    text: str  # cleaned, original casing (dense embedding input)
    lowered: str  # cleaned, lower-cased (sparse input)
    tokens: tuple[str, ...]
    protected: frozenset[str]


def normalize_query(raw: str, *, max_chars: int = 2000) -> NormalizedQuery:
    text = unicodedata.normalize("NFC", raw or "")
    text = _CTRL.sub(" ", text)
    text = _WS.sub(" ", text).strip()[:max_chars].strip()
    if not text:
        raise EmptyQueryError("query is empty after normalisation")
    protected = frozenset(m.group(0).strip("`") for m in PROTECTED.finditer(text))
    tokens = tuple(t for t in _TOKEN.findall(text) if t.strip("'`.:/-"))
    return NormalizedQuery(
        original=raw, text=text, lowered=text.lower(), tokens=tokens, protected=protected
    )
