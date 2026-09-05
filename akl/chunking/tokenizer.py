"""Token counting with the bge-small tokenizer (PRD §4.4).

Uses the exact ``BAAI/bge-small-en-v1.5`` WordPiece tokenizer when it is available
locally (``<models_dir>/bge-small-en-v1.5/tokenizer.json``) or downloadable; falls
back to a deterministic WordPiece-like heuristic (≈1.3 tokens per word, punctuation
split) so chunking and unit tests work offline. The counter in use is exposed as
``TokenCounter.backend``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

_WORD = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]", re.UNICODE)
DEFAULT_MODEL_ID = "BAAI/bge-small-en-v1.5"


class TokenCounter:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        models_dir: Path | None = None,
        *,
        allow_download: bool = False,
    ) -> None:
        self.backend = "heuristic"
        self._tok = None
        local = (models_dir or Path("/models")) / model_id.split("/")[-1] / "tokenizer.json"
        try:
            from tokenizers import Tokenizer

            if local.exists():
                self._tok = Tokenizer.from_file(str(local))
                self.backend = "tokenizers:file"
            elif allow_download:
                self._tok = Tokenizer.from_pretrained(model_id)
                self.backend = "tokenizers:hub"
        except Exception:  # offline, missing package or download failure → heuristic
            self._tok = None
            self.backend = "heuristic"

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._tok is not None:
            return len(self._tok.encode(text, add_special_tokens=False).ids)
        return _heuristic_count(text)

    def count_many(self, texts: Sequence[str]) -> list[int]:
        return [self.count(t) for t in texts]


@lru_cache(maxsize=65536)
def _heuristic_count(text: str) -> int:
    """WordPiece-like estimate: words ≤ 6 chars = 1 token, longer words ≈ 1 per 5 chars, punctuation = 1."""
    total = 0
    for m in _WORD.finditer(text):
        tok = m.group(0)
        if tok.isalpha():
            total += 1 if len(tok) <= 6 else 1 + (len(tok) - 1) // 5
        elif tok.isdigit():
            total += 1 + (len(tok) - 1) // 3
        else:
            total += 1
    return total
