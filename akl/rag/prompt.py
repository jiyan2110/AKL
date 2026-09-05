"""Prompt templates and message assembly (PRD §6.5). Templates live in configs/prompts/*.md and are versioned."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from akl.rag.context_builder import BuiltContext
from akl.rag.llm.provider import Message

DEFAULT_PROMPTS_DIR = Path("configs/prompts")


@dataclass(frozen=True)
class BuiltPrompt:
    messages: list[Message]
    prompt_version: str
    input_tokens: int
    truncated_history: bool


class PromptBuilder:
    def __init__(
        self,
        prompts_dir: Path,
        *,
        org_name: str,
        count_tokens: Callable[[str], int],
        max_input_tokens: int,
    ) -> None:
        self.prompts_dir = prompts_dir
        self.org_name = org_name
        self.count = count_tokens
        self.max_input_tokens = max_input_tokens
        self._cache: dict[str, str] = {}

    def template(self, name: str) -> str:
        if name not in self._cache:
            path = self.prompts_dir / f"{name}.md"
            self._cache[name] = (
                path.read_text(encoding="utf-8") if path.exists() else _BUILTIN[name]
            )
        return self._cache[name]

    def build_answer(
        self,
        question: str,
        context: BuiltContext,
        *,
        summary: str | None = None,
        history: Sequence[Message] = (),
        version: str = "answer_v1",
    ) -> BuiltPrompt:
        system = self.template(version).replace("{org_name}", self.org_name)
        messages: list[Message] = [{"role": "system", "content": system}]
        if summary:
            messages.append(
                {"role": "system", "content": f"Conversation summary so far:\n{summary}"}
            )
        user_block = f"{context.render()}\n\nQuestion: {question}"
        base_tokens = (
            self.count(system) + (self.count(summary) if summary else 0) + self.count(user_block)
        )
        kept: list[Message] = list(history)
        truncated = False
        while (
            kept
            and base_tokens + sum(self.count(m["content"]) for m in kept) > self.max_input_tokens
        ):
            kept.pop(
                0
            )  # drop oldest turns first; context blocks are never dropped here (budgeted upstream)
            truncated = True
        messages.extend(kept)
        messages.append({"role": "user", "content": user_block})
        total = base_tokens + sum(self.count(m["content"]) for m in kept)
        return BuiltPrompt(
            messages=messages,
            prompt_version=version,
            input_tokens=total,
            truncated_history=truncated,
        )

    def build_summary(
        self, existing: str | None, turns: Sequence[Message], *, version: str = "summarize_v1"
    ) -> list[Message]:
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in turns)
        body = (f"Existing summary:\n{existing}\n\n" if existing else "") + f"Turns:\n{transcript}"
        return [
            {"role": "system", "content": self.template(version)},
            {"role": "user", "content": body},
        ]

    def build_rewrite(
        self,
        summary: str | None,
        turns: Sequence[Message],
        latest: str,
        *,
        version: str = "rewrite_v1",
    ) -> list[Message]:
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in turns)
        body = (
            f"Summary:\n{summary}\n\n" if summary else ""
        ) + f"Recent turns:\n{transcript}\n\nLatest message: {latest}"
        return [
            {"role": "system", "content": self.template(version)},
            {"role": "user", "content": body},
        ]


_BUILTIN: dict[str, str] = {
    "answer_v1": (
        "You are the internal knowledge assistant for {org_name}. Answer ONLY from the numbered context blocks. "
        "Cite every factual sentence with [n] markers. If the context lacks the answer reply INSUFFICIENT_EVIDENCE."
    ),
    "summarize_v1": "Summarise the conversation in at most 300 tokens, keeping goals, decisions and cited document titles.",
    "rewrite_v1": "Rewrite the latest message as a standalone query resolving pronouns. Output only the query.",
}
