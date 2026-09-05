"""Cross-encoder reranking (PRD §6.3.4) with an offline lexical fallback.

* :class:`OnnxCrossEncoder` — ``cross-encoder/ms-marco-MiniLM-L-6-v2`` via onnxruntime; scores
  are ``sigmoid(logit)`` ∈ (0, 1). Files (``tokenizer.json`` + ``onnx/model.onnx`` or the
  quantised variant) are downloaded from the Hub into ``<models_dir>`` on first use.
* :class:`LexicalReranker` — deterministic overlap-based scorer used offline/in tests and as
  the degraded mode when the model cannot be loaded (flag ``reranker_fallback``).
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from akl.config import RetrievalSettings
from akl.embedding.bm25.tokenizer import STOPWORDS, tokenize
from akl.embedding.provider import EmbeddingModelError
from akl.rag.retrieval.models import Candidate

_TOKEN = re.compile(r"\w+")


class Reranker(ABC):
    name: str = "base"

    @abstractmethod
    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...

    def rerank(self, query: str, candidates: list[Candidate], *, top_n: int) -> list[Candidate]:
        pool = candidates[:top_n]
        if not pool:
            return []
        scores = self.score(
            query, [f"{c.payload.get('context_prefix') or ''}\n{c.text}".strip() for c in pool]
        )
        for cand, s in zip(pool, scores, strict=True):
            cand.rerank_score = float(s)
        rest = candidates[top_n:]
        return sorted(pool, key=lambda c: -(c.rerank_score or 0.0)) + rest


class LexicalReranker(Reranker):
    """Deterministic fallback scorer, calibrated to the cross-encoder's (0, 1) range.

    Uses the BM25 tokenizer (identifier splitting + stemming) and accepts prefix matches
    of ≥ 5 characters (``postgres`` ↔ ``postgresql``), so scores are comparable to the
    ONNX reranker on lexically matching passages and the shared confidence gate applies.
    """

    name = "lexical"
    _PREFIX_MIN = 5

    @staticmethod
    def _terms(text: str) -> list[str]:
        return [t for t in tokenize(text, keep_stopwords=False) if t not in STOPWORDS]

    @classmethod
    def _matches(cls, term: str, bag: set[str]) -> bool:
        if term in bag:
            return True
        if len(term) >= cls._PREFIX_MIN:
            return any(
                w.startswith(term) or term.startswith(w) for w in bag if len(w) >= cls._PREFIX_MIN
            )
        return False

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        q = list(dict.fromkeys(self._terms(query)))
        out: list[float] = []
        for p in passages:
            words = self._terms(p)
            if not q or not words:
                out.append(0.0)
                continue
            bag = set(words)
            matched = [t for t in q if self._matches(t, bag)]
            coverage = len(matched) / len(q)
            density = (
                sum(1 for w in words if self._matches(w, set(matched))) / len(words)
                if matched
                else 0.0
            )
            out.append(round(min(1.0, 0.75 * coverage + 0.25 * min(1.0, density * 10)), 4))
        return out


class OnnxCrossEncoder(Reranker):
    name = "onnx"

    def __init__(
        self, settings: RetrievalSettings, models_dir: Path, *, allow_download: bool = True
    ) -> None:
        self.model_id = settings.rerank_model_id
        self._dir = models_dir / self.model_id.split("/")[-1]
        self._int8 = settings.rerank_onnx_int8
        self._allow_download = allow_download
        self._session: Any = None
        self._tokenizer: Any = None
        self._input_names: set[str] = set()

    def _load(self) -> None:
        if self._session is not None:
            return
        onnx_name = "onnx/model_quantized.onnx" if self._int8 else "onnx/model.onnx"
        tok, model = self._dir / "tokenizer.json", self._dir / onnx_name
        if not (tok.exists() and model.exists()):
            if not self._allow_download:
                raise EmbeddingModelError(
                    "reranker files missing and download disabled", details={"dir": str(self._dir)}
                )
            try:
                from huggingface_hub import hf_hub_download

                self._dir.mkdir(parents=True, exist_ok=True)
                for filename in ("tokenizer.json", onnx_name):
                    hf_hub_download(
                        repo_id=self.model_id, filename=filename, local_dir=str(self._dir)
                    )
            except Exception as exc:
                raise EmbeddingModelError(
                    "failed to download reranker",
                    details={"model": self.model_id, "error": str(exc)},
                ) from exc
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(tok))
            tokenizer.enable_truncation(512)
            tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")  # noqa: S106 - tokenizer symbol
            self._session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
            self._tokenizer = tokenizer
            self._input_names = {i.name for i in self._session.get_inputs()}
        except Exception as exc:
            raise EmbeddingModelError(
                "failed to load reranker ONNX model", details={"error": str(exc)}
            ) from exc

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self._load()
        encoded = self._tokenizer.encode_batch([(query, p) for p in passages])
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feeds: dict[str, np.ndarray] = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array([e.type_ids for e in encoded], dtype=np.int64)
        logits = np.asarray(self._session.run(None, feeds)[0], dtype=np.float64).reshape(
            len(passages), -1
        )[:, 0]
        return [float(1.0 / (1.0 + math.exp(-x))) for x in logits]


def build_reranker(
    settings: RetrievalSettings, models_dir: Path, *, allow_download: bool = True
) -> Reranker | None:
    if not settings.rerank_enabled or settings.rerank_provider == "none":
        return None
    if settings.rerank_provider == "lexical":
        return LexicalReranker()
    return OnnxCrossEncoder(settings, models_dir, allow_download=allow_download)
