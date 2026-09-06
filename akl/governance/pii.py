"""PII scanning (PRD §9.6): detect likely-personal data in ingested text.

Deliberately conservative and regex-based — no ML model, no external service, and the raw
matched value never leaves this module: callers get a :class:`PiiFinding` with only the type
and a truncated positional hint; the value itself is hashed by the caller before it is ever
persisted (see :mod:`akl.db.repositories.pii`). False positives are expected and acceptable —
this powers a *flag for review*, not an automated redaction or block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)")
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_CREDIT_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)")
_IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", _EMAIL),
    ("phone", _PHONE),
    ("ssn", _SSN),
    ("credit_card", _CREDIT_CARD),
    ("ip_address", _IPV4),
)


def _luhn_ok(digits: str) -> bool:
    """Luhn checksum — cuts credit-card false positives (version strings, phone numbers, ids)."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass(frozen=True)
class PiiFinding:
    pii_type: str
    value: str  # kept only in-process; never logged or persisted raw (see db/repositories/pii.py)
    start: int
    end: int


def scan_text(text: str, *, enabled_types: frozenset[str] | None = None) -> list[PiiFinding]:
    """Scan ``text`` for the enabled PII types (default: all). Deduplicates identical values."""
    seen: set[tuple[str, str]] = set()
    out: list[PiiFinding] = []
    for pii_type, pattern in _PATTERNS:
        if enabled_types is not None and pii_type not in enabled_types:
            continue
        for m in pattern.finditer(text):
            value = m.group(0)
            if pii_type == "credit_card":
                digits = re.sub(r"[ -]", "", value)
                if len(digits) < 13 or len(digits) > 19 or not _luhn_ok(digits):
                    continue
                value = digits
            key = (pii_type, value)
            if key in seen:
                continue
            seen.add(key)
            out.append(PiiFinding(pii_type, value, m.start(), m.end()))
    return out
