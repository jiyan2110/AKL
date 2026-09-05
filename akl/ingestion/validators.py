"""Document validation rules (PRD §3.5) and secret detection.

Rule evaluation order is by *specificity*: content diagnostics (AKL-E3004 binary,
AKL-E3005 insufficient text) precede the coarse byte-size bound (AKL-E3001), so a
19-byte Markdown file is quarantined for what is actually wrong with it — too
little text — while a 60 MiB upload is still rejected for size before parsing.
``ValidationResult.reject_code`` is the first (most specific) rejecting rule; every
violated rule is still recorded in ``detail``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from akl.ingestion.models import UnifiedDocument

MIN_DOC_BYTES = 64
MAX_DOC_BYTES = 50 * 1024 * 1024
MIN_TEXT_CHARS = 100
MIN_TEXT_DENSITY = 0.4
MAX_HEADING_DEPTH = 8

SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "aws_access_key": re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "slack_token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    "generic_api_key": re.compile(
        r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{20,}"
    ),
}


@dataclass(frozen=True)
class Violation:
    code: str
    severity: str  # "reject" | "warn"
    message: str


@dataclass
class ValidationResult:
    violations: list[Violation] = field(default_factory=list)

    @property
    def rejected(self) -> bool:
        return any(v.severity == "reject" for v in self.violations)

    @property
    def reject_code(self) -> str | None:
        return next((v.code for v in self.violations if v.severity == "reject"), None)

    @property
    def flags(self) -> tuple[str, ...]:
        return tuple(
            v.code.lower().replace("-", "_") for v in self.violations if v.severity == "warn"
        )

    @property
    def detail(self) -> str:
        return "; ".join(f"{v.code}: {v.message}" for v in self.violations)


def text_density(text: str) -> float:
    if not text:
        return 0.0
    return sum(ch.isalnum() or ch.isspace() for ch in text) / len(text)


def find_secrets(text: str) -> list[str]:
    return [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(text)]


def validate_size_upper(size: int, *, max_bytes: int = MAX_DOC_BYTES) -> ValidationResult:
    """Pre-parse guard: oversized objects are never parsed (AKL-E3001)."""
    result = ValidationResult()
    if size > max_bytes:
        result.violations.append(Violation("AKL-E3001", "reject", f"size {size} > {max_bytes}"))
    return result


def validate_bytes(
    size: int, *, min_bytes: int = MIN_DOC_BYTES, max_bytes: int = MAX_DOC_BYTES
) -> ValidationResult:
    """Full byte-size rule (both bounds); used standalone and by :func:`validate_document`."""
    result = ValidationResult()
    if size < min_bytes or size > max_bytes:
        result.violations.append(
            Violation("AKL-E3001", "reject", f"size {size} outside [{min_bytes}, {max_bytes}]")
        )
    return result


def validate_document(
    doc: UnifiedDocument,
    *,
    size_bytes: int | None = None,
    allow_secret_like: bool = False,
    min_text_chars: int = MIN_TEXT_CHARS,
    min_density: float = MIN_TEXT_DENSITY,
    min_bytes: int = MIN_DOC_BYTES,
) -> ValidationResult:
    """Content rules first (specific), then the lower byte bound (coarse)."""
    result = ValidationResult()
    if "\x00" in doc.text or (doc.text and doc.text.count("\ufffd") / max(1, len(doc.text)) > 0.01):
        result.violations.append(
            Violation("AKL-E3004", "reject", "binary or badly encoded content")
        )
    if len(doc.text.strip()) < min_text_chars:
        result.violations.append(
            Violation(
                "AKL-E3005", "reject", f"text has {len(doc.text.strip())} chars < {min_text_chars}"
            )
        )
    if text_density(doc.text) < min_density:
        result.violations.append(Violation("AKL-W3006", "warn", "low text density"))
    secrets = find_secrets(doc.text)
    if secrets:
        severity = "warn" if allow_secret_like else "reject"
        result.violations.append(
            Violation(
                "AKL-E3008" if severity == "reject" else "AKL-W3008",
                severity,
                f"secret-like content: {', '.join(secrets)}",
            )
        )
    if _max_depth(doc) > MAX_HEADING_DEPTH:
        result.violations.append(Violation("AKL-W3014", "warn", "heading depth > 8"))
    if size_bytes is not None and size_bytes < min_bytes:
        result.violations.append(
            Violation("AKL-E3001", "reject", f"size {size_bytes} < {min_bytes}")
        )
    return result


def _max_depth(doc: UnifiedDocument) -> int:
    def depth(nodes: list, d: int = 1) -> int:  # type: ignore[type-arg]
        return max([depth(n.children, d + 1) for n in nodes] + [d]) if nodes else d - 1

    return depth(list(doc.structure))
