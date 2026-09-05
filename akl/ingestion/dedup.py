"""Near-duplicate detection with 64-bit SimHash (PRD §2.5.3, §3.4.8).

Features are word 4-gram shingles weighted by their frequency (Charikar's SimHash).
Frequency weighting stabilises fingerprint bits for short documents, where an
unweighted set-based SimHash flips too many bits on a handful of edits; word
4-grams keep unrelated documents drawn from a small vocabulary far apart.
Calibration (deterministic, blake2b): a 600-word document with 2 word edits →
distance 2; unrelated 600-word documents from the same 12-word vocabulary → 28.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

_TOKEN = re.compile(r"\w+", re.UNICODE)
HAMMING_MAX = 3
SHINGLE_SIZE = 4
MASK64 = (1 << 64) - 1


def shingles(text: str, n: int = SHINGLE_SIZE) -> Iterable[str]:
    tokens = [t.lower() for t in _TOKEN.findall(text)]
    if len(tokens) < n:
        yield " ".join(tokens)
        return
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i : i + n])


def simhash(text: str) -> int:
    """64-bit frequency-weighted SimHash; unsigned integer suitable for a ``uint64`` column."""
    weights = Counter(sh for sh in shingles(text) if sh)
    if not weights:
        return 0
    vector = [0] * 64
    for shingle, weight in weights.items():
        h = int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            vector[bit] += weight if (h >> bit) & 1 else -weight
    out = 0
    for bit in range(64):
        if vector[bit] > 0:
            out |= 1 << bit
    return out & MASK64


def hamming(a: int, b: int) -> int:
    return bin((a ^ b) & MASK64).count("1")


@dataclass(frozen=True)
class DuplicateDecision:
    duplicate_document_id: str
    canonical_document_id: str
    hamming_distance: int
    fingerprint: int


def find_near_duplicates(
    candidates: list[tuple[str, int, float]],
    existing: list[tuple[str, int, float]],
    *,
    max_distance: int = HAMMING_MAX,
) -> list[DuplicateDecision]:
    """Decide duplicates among ``candidates`` against ``existing`` (and each other).

    Tuples are ``(document_id, fingerprint, quality_score)``. The canonical document
    is the earlier-ingested one — an ``existing`` document, or the candidate that comes
    first in ``candidates`` order — unless the later one's quality exceeds it by ≥ 0.1
    (PRD §2.5.3). Callers pass candidates in a deterministic order (the service uses
    canonical URI order within a run).
    """
    decisions: list[DuplicateDecision] = []
    pool: list[tuple[str, int, float]] = list(existing)
    for doc_id, fp, quality in candidates:
        if fp == 0:
            pool.append((doc_id, fp, quality))
            continue
        match = next(
            (
                (oid, ofp, oq)
                for oid, ofp, oq in pool
                if oid != doc_id and ofp and hamming(fp, ofp) <= max_distance
            ),
            None,
        )
        if match is None:
            pool.append((doc_id, fp, quality))
            continue
        other_id, other_fp, other_q = match
        if quality >= other_q + 0.1:
            decisions.append(DuplicateDecision(other_id, doc_id, hamming(fp, other_fp), other_fp))
            pool = [(i, f, q) for i, f, q in pool if i != other_id] + [(doc_id, fp, quality)]
        else:
            decisions.append(DuplicateDecision(doc_id, other_id, hamming(fp, other_fp), fp))
    return decisions
