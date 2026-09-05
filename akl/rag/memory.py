"""Conversation memory and multi-turn query rewrite (PRD §6.2.6, §6.8, §6.10)."""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from akl.config import LLMSettings
from akl.db.repositories.conversations import ConversationRepository
from akl.db.session import Database
from akl.errors import AKLError
from akl.rag.llm.provider import LLMProvider, LLMUnavailableError, Message
from akl.rag.prompt import PromptBuilder

_ANAPHORA = re.compile(
    r"\b(it|its|that|this|those|these|the (?:second|first|third|last|other) (?:one|option)|them|he|she|they)\b",
    re.I,
)
_ENTITY = re.compile(
    r"`[^`]+`|\bAKL-[EW]\d{4}\b|\b[A-Z][A-Z0-9_]{3,}\b|\b[a-z]+(?:_[a-z0-9]+)+\b|\b\w+\.(?:py|md|yaml|yml|json|sql)\b"
)


@dataclass
class MemoryState:
    conversation_id: uuid.UUID
    summary: str | None
    history: list[Message] = field(default_factory=list)  # last N turns (user/assistant)
    last_citation_chunk_ids: list[str] = field(default_factory=list)
    created: bool = False


class ConversationMemory:
    def __init__(
        self,
        db: Database,
        settings: LLMSettings,
        prompt: PromptBuilder,
        llm: LLMProvider | None,
        count_tokens: Callable[[str], int],
    ) -> None:
        self.db = db
        self.settings = settings
        self.prompt = prompt
        self.llm = llm
        self.count = count_tokens

    # -- load / create ---------------------------------------------------------------------------
    def load(self, conversation_id: uuid.UUID | None, principal_id: str) -> MemoryState:
        with self.db.session() as s:
            repo = ConversationRepository(s)
            if conversation_id is None:
                created = repo.create(principal_id, ttl_days=self.settings.conversation_ttl_days)
                return MemoryState(created.conversation_id, None, [], [], created=True)
            conv = repo.get_owned(conversation_id, principal_id)
            if conv is None:
                raise ConversationNotFoundError(conversation_id)
            turns = repo.messages(conv.conversation_id, limit=self.settings.rag_history_turns * 2)
            history: list[Message] = [{"role": m.role, "content": m.content} for m in turns]
            last_cites: list[str] = []
            for m in reversed(turns):
                if m.role == "assistant":
                    last_cites = [str(c) for c in repo.citation_chunk_ids(m.message_id)]
                    break
            return MemoryState(conv.conversation_id, conv.summary, history, last_cites)

    # -- rewrite ----------------------------------------------------------------------------------
    def rewrite(self, query: str, state: MemoryState) -> str:
        """Standalone query for retrieval; LLM-based when available, rule-based fallback (PRD §6.2.6)."""
        needs = bool(state.history) and (
            len(query.split()) < 4 or _ANAPHORA.search(query) is not None
        )
        if not needs:
            return query
        if self.llm is not None:
            try:
                msgs = self.prompt.build_rewrite(
                    state.summary, state.history[-self.settings.rag_history_turns * 2 :], query
                )
                out = (
                    self.llm.complete(msgs, max_tokens=64, temperature=0.0)
                    .text.strip()
                    .splitlines()
                )
                if out and out[0].strip():
                    return out[0].strip()
            except LLMUnavailableError:
                pass
        # rule-based: append the most recent entity mentioned by the user
        for m in reversed(state.history):
            if m["role"] != "user":
                continue
            ents = _ENTITY.findall(m["content"])
            if ents:
                return f"{query} {ents[-1].strip('`')}"
        return f"{query} {state.history[-1]['content'][:80]}" if state.history else query

    # -- persist ---------------------------------------------------------------------------------
    def record(
        self,
        state: MemoryState,
        *,
        question: str,
        rewritten: str,
        answer: str | None,
        mode: str,
        confidence: float,
        flags: Sequence[str],
        citations: Sequence[dict[str, Any]],
        trace_id: str,
        request_id: str,
    ) -> None:
        with self.db.session() as s:
            repo = ConversationRepository(s)
            conv = repo.get(state.conversation_id)
            if conv is None:
                return
            repo.add_turn(
                conv,
                role="user",
                content=question,
                rewritten_query=rewritten,
                token_count=self.count(question),
                request_id=request_id,
            )
            repo.add_turn(
                conv,
                role="assistant",
                content=answer or "INSUFFICIENT_EVIDENCE",
                token_count=self.count(answer or ""),
                trace_id=trace_id,
                request_id=request_id,
                mode=mode,
                confidence=confidence,
                flags=flags,
                citations=list(citations),
            )
            self._maybe_summarise(repo, conv)

    def _maybe_summarise(self, repo: ConversationRepository, conv: object) -> None:
        """Fold older turns into ``summary`` once the un-summarised history exceeds the trigger (PRD §6.8)."""
        from akl.db.models import Conversation

        assert isinstance(conv, Conversation)
        turns = repo.messages(conv.conversation_id)
        keep = self.settings.rag_history_turns * 2
        older = turns[:-keep] if len(turns) > keep else []
        pending = (
            sum(m.token_count or 0 for m in older if m.turn > conv.summary_tokens_turn)
            if hasattr(conv, "summary_tokens_turn")
            else sum(m.token_count or 0 for m in older)
        )
        if not older or pending < self.settings.rag_summary_trigger_tokens:
            return
        msgs = [{"role": m.role, "content": m.content} for m in older]
        summary: str
        if self.llm is not None:
            try:
                summary = self.llm.complete(
                    self.prompt.build_summary(conv.summary, msgs), max_tokens=400, temperature=0.0
                ).text.strip()
            except LLMUnavailableError:
                summary = _rule_summary(conv.summary, msgs)
        else:
            summary = _rule_summary(conv.summary, msgs)
        repo.set_summary(conv, summary, self.count(summary))


def _rule_summary(existing: str | None, turns: Sequence[Message]) -> str:
    """No-LLM summary: previous summary + the user's questions (PRD §6.8 fallback)."""
    questions = [m["content"] for m in turns if m["role"] == "user"]
    parts = [existing] if existing else []
    parts.append("Earlier questions: " + " | ".join(q[:120] for q in questions))
    return "\n".join(parts)[:2000]


class ConversationNotFoundError(AKLError):
    """Conversation missing or owned by another principal (AKL-E6030)."""

    code = "AKL-E6030"
    http_status = 404
    retryable = False

    def __init__(self, conversation_id: uuid.UUID) -> None:
        super().__init__(
            "conversation not found", details={"conversation_id": str(conversation_id)}
        )
        self.conversation_id = conversation_id
