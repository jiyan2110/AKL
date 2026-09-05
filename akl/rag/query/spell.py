"""Corpus-vocabulary spell correction (PRD §6.2.2).

Only tokens absent from the corpus vocabulary and not protected are corrected, to a
vocabulary word within edit distance ≤ 2 (best by frequency). No external dictionary.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from collections.abc import Iterable

_WORD = re.compile(r"[a-z][a-z0-9]{2,}")


class SpellCorrector:
    def __init__(
        self, vocabulary: Counter[str] | Iterable[str] | None = None, *, min_freq: int = 2
    ) -> None:
        counts = vocabulary if isinstance(vocabulary, Counter) else Counter(vocabulary or [])
        self.vocab: Counter[str] = Counter(
            {w: n for w, n in counts.items() if n >= min_freq and _WORD.fullmatch(w)}
        )
        self._by_prefix: dict[str, list[str]] = {}
        for word in self.vocab:
            self._by_prefix.setdefault(word[0], []).append(word)

    @classmethod
    def from_texts(cls, texts: Iterable[str], **kw: int) -> SpellCorrector:
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(_WORD.findall(text.lower()))
        return cls(counts, **kw)

    def correct_token(self, token: str) -> str | None:
        low = token.lower()
        if not _WORD.fullmatch(low) or low in self.vocab or len(low) < 4:
            return None
        candidates = self._by_prefix.get(low[0], [])
        close = difflib.get_close_matches(low, candidates, n=5, cutoff=0.8)
        close = [c for c in close if abs(len(c) - len(low)) <= 2]
        if not close:
            return None
        return max(close, key=lambda c: (self.vocab[c], -abs(len(c) - len(low))))

    def correct(
        self, text: str, protected: frozenset[str] = frozenset()
    ) -> tuple[str, dict[str, str]]:
        corrections: dict[str, str] = {}
        out: list[str] = []
        for token in text.split(" "):
            core = token.strip(".,;:!?()[]\"'")
            if not core or core in protected or core.lower() in protected:
                out.append(token)
                continue
            fixed = self.correct_token(core)
            if fixed:
                corrections[core] = fixed
                out.append(token.replace(core, fixed))
            else:
                out.append(token)
        return " ".join(out), corrections

    def __len__(self) -> int:
        return len(self.vocab)
