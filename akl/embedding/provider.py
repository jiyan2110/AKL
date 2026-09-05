"""Embedding provider contract and implementations (PRD §5.3).

* :class:`BgeOnnxProvider` — ``BAAI/bge-small-en-v1.5`` via onnxruntime (CLS pooling, L2 norm).
  Model files are downloaded from the Hugging Face Hub into ``<models_dir>/bge-small-en-v1.5``
  on first use (``tokenizer.json`` + ``onnx/model.onnx``).
* :class:`HashEmbeddingProvider` — deterministic, dependency-free 384-d vectors from hashed
  word/character n-grams. Used for offline tests and ``AKL_EMBED_PROVIDER=hash``; it preserves
  lexical similarity (shared n-grams → higher cosine) so retrieval tests are meaningful.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from akl.config import EmbeddingSettings
from akl.errors import AKLError

_TOKEN = re.compile(r"\w+", re.UNICODE)


class EmbeddingModelError(AKLError):
    """Model download/load failure (AKL-E5001)."""

    code = "AKL-E5001"
    retryable = True


class EmbeddingError(AKLError):
    """Inference failure (AKL-E5002)."""

    code = "AKL-E5002"
    retryable = True


class EmbeddingProvider(ABC):
    """Asymmetric embedding interface: documents vs queries (PRD §5.3)."""

    model_id: str
    model_version: str
    dim: int
    query_instruction: str = ""

    @property
    def embedding_version(self) -> str:
        return f"{self.model_id.split('/')[-1]}__{self.model_version}__{self.dim}"

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n, dim)`` float32 array of L2-normalised vectors."""

    def embed_query(self, text: str) -> np.ndarray:
        vec: np.ndarray = self.embed_documents([f"{self.query_instruction}{text}"])[0]
        return vec

    def warm_up(self) -> None:
        self.embed_documents(["warm up"])


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalised: np.ndarray = (matrix / norms).astype(np.float32)
    return normalised


# ---------------------------------------------------------------------------
# Hash provider (offline / tests)
# ---------------------------------------------------------------------------
class HashEmbeddingProvider(EmbeddingProvider):
    model_id = "akl/hash-embed"
    model_version = "1"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = [w.lower() for w in _TOKEN.findall(text)]
        feats = list(words)
        feats += [" ".join(words[i : i + 2]) for i in range(len(words) - 1)]
        for w in words:
            padded = f"#{w}#"
            feats += [padded[i : i + 3] for i in range(len(padded) - 2)]
        return feats

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "big") % self.dim
                sign = 1.0 if h[4] & 1 else -1.0
                weight = 1.0 if " " in feat or len(feat) > 3 else 0.5
                out[row, idx] += sign * weight
        return _normalize(out)


# ---------------------------------------------------------------------------
# BGE via onnxruntime
# ---------------------------------------------------------------------------
class BgeOnnxProvider(EmbeddingProvider):
    query_instruction = "Represent this sentence for searching relevant passages: "

    def __init__(
        self, settings: EmbeddingSettings, models_dir: Path, *, allow_download: bool = True
    ) -> None:
        self.model_id = settings.embed_model_id
        self.model_version = settings.embed_model_version
        self.dim = settings.embed_dim
        self.query_instruction = settings.embed_query_instruction
        self.batch_size = settings.embed_batch_size
        self.max_length = 512
        self._dir = models_dir / self.model_id.split("/")[-1]
        self._session: Any = None
        self._tokenizer: Any = None
        self._input_names: set[str] = set()
        self._settings = settings
        self._allow_download = allow_download

    # -- files -----------------------------------------------------------------------
    def ensure_files(self) -> tuple[Path, Path]:
        tok = self._dir / "tokenizer.json"
        onnx_name = (
            "onnx/model_quantized.onnx" if self._settings.embed_onnx_int8 else "onnx/model.onnx"
        )
        model = self._dir / onnx_name
        if tok.exists() and model.exists():
            return tok, model
        if not self._allow_download:
            raise EmbeddingModelError(
                "model files missing and download disabled", details={"dir": str(self._dir)}
            )
        try:
            from huggingface_hub import hf_hub_download

            self._dir.mkdir(parents=True, exist_ok=True)
            for filename in ("tokenizer.json", onnx_name):
                hf_hub_download(repo_id=self.model_id, filename=filename, local_dir=str(self._dir))
        except Exception as exc:
            raise EmbeddingModelError(
                "failed to download embedding model",
                details={"model": self.model_id, "error": str(exc)},
            ) from exc
        if self._settings.embed_model_sha256:
            digest = hashlib.sha256(model.read_bytes()).hexdigest()
            if digest != self._settings.embed_model_sha256:
                raise EmbeddingModelError(
                    "model checksum mismatch (AKL-E5005)",
                    details={"expected": self._settings.embed_model_sha256, "actual": digest},
                )
        return tok, model

    def _load(self) -> None:
        if self._session is not None:
            return
        tok_path, model_path = self.ensure_files()
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(tok_path))
            tokenizer.enable_truncation(self.max_length)
            tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")  # noqa: S106 - tokenizer symbol
            opts = ort.SessionOptions()
            if self._settings.embed_threads:
                opts.intra_op_num_threads = self._settings.embed_threads
            providers = ["CPUExecutionProvider"]
            if (
                self._settings.embed_device in ("auto", "cuda")
                and "CUDAExecutionProvider" in ort.get_available_providers()
            ):
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._session = ort.InferenceSession(
                str(model_path), sess_options=opts, providers=providers
            )
            self._tokenizer = tokenizer
            self._input_names = {i.name for i in self._session.get_inputs()}
        except Exception as exc:
            raise EmbeddingModelError(
                "failed to load ONNX model", details={"error": str(exc)}
            ) from exc

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            encoded = self._tokenizer.encode_batch(batch)
            ids = np.array([e.ids for e in encoded], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            try:
                hidden = self._session.run(None, feeds)[0]
            except Exception as exc:
                raise EmbeddingError(
                    "onnx inference failed", details={"batch": len(batch), "error": str(exc)}
                ) from exc
            cls = hidden[:, 0, :]  # BGE uses CLS pooling
            if not np.all(np.isfinite(cls)):
                raise EmbeddingError("non-finite values in embeddings (AKL-E5004)")
            out.append(cls.astype(np.float32))
        return _normalize(np.vstack(out)) if out else np.zeros((0, self.dim), dtype=np.float32)


def build_provider(
    settings: EmbeddingSettings, models_dir: Path, *, allow_download: bool = True
) -> EmbeddingProvider:
    if settings.embed_provider == "hash":
        return HashEmbeddingProvider(settings.embed_dim)
    if settings.embed_provider == "bge":
        return BgeOnnxProvider(settings, models_dir, allow_download=allow_download)
    raise EmbeddingModelError(
        f"unknown embedding provider {settings.embed_provider!r}", retryable=False
    )


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) or 1.0) * (np.linalg.norm(b) or 1.0)))


def vector_to_bytes(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype="<f4").tobytes()


def bytes_to_vector(data: bytes, dim: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype="<f4")
    if arr.shape[0] != dim:
        raise EmbeddingError(
            "cached vector has wrong dimension",
            details={"expected": dim, "actual": int(arr.shape[0])},
            retryable=False,
        )
    return arr.copy()


def unit_norm_ok(vector: np.ndarray, tol: float = 1e-3) -> bool:
    return math.isclose(float(np.linalg.norm(vector)), 1.0, abs_tol=tol)
