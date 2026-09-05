"""LLM provider abstraction (PRD §6.5–6.9, ADR-010): OpenAI-compatible HTTP or extractive fallback."""

from akl.rag.llm.provider import LLMProvider, LLMResult, LLMUnavailableError, build_llm

__all__ = ["LLMProvider", "LLMResult", "LLMUnavailableError", "build_llm"]
