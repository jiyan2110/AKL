"""ConversationRepository — turns, citations and summaries (PRD §6.8, Appendix A.12)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete, func, select

from akl.db.models import AnswerCitation, Conversation, Message
from akl.db.repositories import Repository


class ConversationRepository(Repository):
    def create(self, principal_id: str, *, ttl_days: int) -> Conversation:
        conv = Conversation(
            conversation_id=uuid.uuid4(),
            principal_id=principal_id,
            expires_at=datetime.now(UTC) + timedelta(days=ttl_days),
        )
        self.session.add(conv)
        self.session.flush()
        return conv

    def get(self, conversation_id: uuid.UUID) -> Conversation | None:
        return self.session.get(Conversation, conversation_id)

    def get_owned(self, conversation_id: uuid.UUID, principal_id: str) -> Conversation | None:
        conv = self.get(conversation_id)
        return conv if conv is not None and conv.principal_id == principal_id else None

    def messages(self, conversation_id: uuid.UUID, *, limit: int | None = None) -> list[Message]:
        stmt = (
            select(Message).where(Message.conversation_id == conversation_id).order_by(Message.turn)
        )
        rows = list(self.session.scalars(stmt))
        return rows[-limit:] if limit else rows

    def add_turn(
        self,
        conv: Conversation,
        *,
        role: str,
        content: str,
        rewritten_query: str | None = None,
        token_count: int | None = None,
        trace_id: str | None = None,
        request_id: str | None = None,
        mode: str | None = None,
        confidence: float | None = None,
        flags: Sequence[str] = (),
        citations: Sequence[dict[str, Any]] = (),
    ) -> Message:
        conv.turn_count += 1
        conv.updated_at = datetime.now(UTC)
        msg = Message(
            message_id=uuid.uuid4(),
            conversation_id=conv.conversation_id,
            turn=conv.turn_count,
            role=role,
            content=content,
            rewritten_query=rewritten_query,
            token_count=token_count,
            trace_id=trace_id,
            request_id=request_id,
            mode=mode,
            confidence=confidence,
            flags=list(flags),
        )
        self.session.add(msg)
        self.session.flush()
        for c in citations:
            self.session.add(
                AnswerCitation(
                    message_id=msg.message_id,
                    citation_index=int(c["index"]),
                    chunk_id=_uuid(c.get("chunk_id")),
                    lineage_id=_uuid(c.get("lineage_id")),
                    document_id=_uuid(c.get("document_id")),
                    locator=c.get("locator"),
                    score=c.get("score"),
                )
            )
        self.session.flush()
        return msg

    def citation_chunk_ids(self, message_id: uuid.UUID) -> list[uuid.UUID]:
        stmt = (
            select(AnswerCitation.chunk_id)
            .where(AnswerCitation.message_id == message_id)
            .order_by(AnswerCitation.citation_index)
        )
        return [c for c in self.session.scalars(stmt) if c is not None]

    def set_summary(self, conv: Conversation, summary: str, tokens: int) -> None:
        conv.summary = summary
        conv.summary_tokens = tokens

    def unsummarised_tokens(self, conv: Conversation, since_turns: int) -> int:
        stmt = select(func.coalesce(func.sum(Message.token_count), 0)).where(
            Message.conversation_id == conv.conversation_id, Message.turn > since_turns
        )
        return int(self.session.scalar(stmt) or 0)

    def delete(self, conversation_id: uuid.UUID) -> None:
        self.session.execute(
            delete(Conversation).where(Conversation.conversation_id == conversation_id)
        )

    def purge_expired(self) -> int:
        result = self.session.execute(
            delete(Conversation).where(Conversation.expires_at < datetime.now(UTC))
        )
        return int(getattr(result, "rowcount", 0) or 0)


def _uuid(value: Any) -> uuid.UUID | None:
    if not value:
        return None
    try:
        return uuid.UUID(str(value))
    except ValueError:
        return None
