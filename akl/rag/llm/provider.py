"""LLM provider contract and the OpenAI-compatible implementation (PRD §6.5, §6.9, Appendix B.9)."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from akl.config import LLMSettings
from akl.errors import AKLError

Message = dict[str, str]  # {"role": ..., "content": ...}


class LLMUnavailableError(AKLError):
    """Generation backend timed out / errored (AKL-W6020 → extractive fallback)."""

    code = "AKL-W6020"
    http_status = 503
    retryable = True


class LLMConfigError(AKLError):
    code = "AKL-E6021"
    http_status = 502
    retryable = False


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    first_token_ms: float | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def cost_usd(self, price_in: float, price_out: float) -> float:
        return round(self.input_tokens / 1000 * price_in + self.output_tokens / 1000 * price_out, 6)


class LLMProvider(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    def complete(
        self, messages: Sequence[Message], *, max_tokens: int, temperature: float
    ) -> LLMResult: ...

    @abstractmethod
    def stream(
        self, messages: Sequence[Message], *, max_tokens: int, temperature: float
    ) -> Iterator[str]:
        """Yield text deltas; the final assembled text is the concatenation."""

    def available(self) -> bool:
        return True


class OpenAICompatProvider(LLMProvider):
    """Any server implementing ``POST /v1/chat/completions`` (Ollama, llama.cpp, vLLM, OpenAI)."""

    name = "openai_compat"

    def __init__(
        self, settings: LLMSettings, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        if not settings.llm_model:
            raise LLMConfigError("AKL_LLM_MODEL must be set for the openai_compat provider")
        self.settings = settings
        self.model = settings.llm_model
        headers = {"Content-Type": "application/json"}
        if settings.llm_api_key:
            headers["Authorization"] = f"Bearer {settings.llm_api_key.get_secret_value()}"
        self._client = httpx.Client(
            base_url=settings.llm_base_url.rstrip("/"),
            headers=headers,
            timeout=settings.llm_timeout_s,
            transport=transport,
        )

    def _payload(
        self, messages: Sequence[Message], max_tokens: int, temperature: float, stream: bool
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

    def complete(
        self, messages: Sequence[Message], *, max_tokens: int, temperature: float
    ) -> LLMResult:
        start = time.perf_counter()
        try:
            resp = self._client.post(
                "/chat/completions", json=self._payload(messages, max_tokens, temperature, False)
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(
                f"llm request failed: {exc}", details={"base_url": self.settings.llm_base_url}
            ) from exc
        except ValueError as exc:
            raise LLMUnavailableError("llm returned non-JSON", details={"error": str(exc)}) from exc
        try:
            choice = data["choices"][0]
            text = str(choice["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailableError(
                "unexpected llm response shape",
                details={"keys": list(data) if isinstance(data, dict) else []},
            ) from exc
        usage = data.get("usage") or {}
        return LLMResult(
            text=text,
            model=str(data.get("model") or self.model),
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            finish_reason=choice.get("finish_reason"),
            raw={"id": data.get("id")},
        )

    def stream(
        self, messages: Sequence[Message], *, max_tokens: int, temperature: float
    ) -> Iterator[str]:
        try:
            with self._client.stream(
                "POST",
                "/chat/completions",
                json=self._payload(messages, max_tokens, temperature, True),
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                    except ValueError:
                        continue
                    for choice in obj.get("choices", []):
                        delta = (choice.get("delta") or {}).get("content")
                        if delta:
                            yield str(delta)
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(
                f"llm stream failed: {exc}", details={"base_url": self.settings.llm_base_url}
            ) from exc

    def available(self) -> bool:
        try:
            return self._client.get("/models").status_code < 500
        except httpx.HTTPError:
            return False


def build_llm(
    settings: LLMSettings, *, transport: httpx.BaseTransport | None = None
) -> LLMProvider | None:
    """``None`` means: no generation model configured → extractive mode (ADR-010)."""
    if settings.llm_provider == "none":
        return None
    if settings.llm_provider == "openai_compat":
        return OpenAICompatProvider(settings, transport=transport)
    raise LLMConfigError(f"unknown llm provider {settings.llm_provider!r}")
