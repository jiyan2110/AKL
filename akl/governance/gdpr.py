"""GDPR erasure and export for a principal's own data (PRD §9.7).

AKL's only per-principal personal data is conversation history (turns + citations); ingested
documents belong to the organisation, not to the principal who uploaded them, so erasure here is
scoped to conversations. ``erase_principal`` and ``export_principal`` both operate within a single
transaction and return a plain summary dict (audit-loggable, JSON-serialisable).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from akl.db.models import AnswerCitation, Conversation, Message


@dataclass
class ErasureResult:
    principal_id: str
    conversations_deleted: int
    messages_deleted: int
    citations_deleted: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "principal_id": self.principal_id,
            "conversations_deleted": self.conversations_deleted,
            "messages_deleted": self.messages_deleted,
            "citations_deleted": self.citations_deleted,
        }


def erase_principal(session: Session, principal_id: str) -> ErasureResult:
    """Delete every conversation (and its messages/citations, via FK cascade) owned by ``principal_id``."""
    conv_ids = list(
        session.scalars(
            select(Conversation.conversation_id).where(Conversation.principal_id == principal_id)
        )
    )
    if not conv_ids:
        return ErasureResult(principal_id, 0, 0, 0)
    message_ids = list(
        session.scalars(select(Message.message_id).where(Message.conversation_id.in_(conv_ids)))
    )
    citations = 0
    if message_ids:
        citations = len(
            list(
                session.scalars(
                    select(AnswerCitation.id).where(AnswerCitation.message_id.in_(message_ids))
                )
            )
        )
    messages = len(message_ids)
    for conv_id in conv_ids:
        conv = session.get(Conversation, conv_id)
        if conv is not None:
            session.delete(conv)  # ON DELETE CASCADE removes messages -> answer_citations
    return ErasureResult(principal_id, len(conv_ids), messages, citations)


def export_principal(
    session: Session, principal_id: str, *, max_conversations: int = 500
) -> dict[str, Any]:
    """Everything AKL holds about ``principal_id``: their conversations, turns, and citations."""
    conv_stmt = (
        select(Conversation)
        .where(Conversation.principal_id == principal_id)
        .order_by(Conversation.created_at.desc())
        .limit(max_conversations)
    )
    conversations = list(session.scalars(conv_stmt))
    out: list[dict[str, Any]] = []
    for conv in conversations:
        turns = list(
            session.scalars(
                select(Message)
                .where(Message.conversation_id == conv.conversation_id)
                .order_by(Message.turn)
            )
        )
        turn_payload = []
        for m in turns:
            citations = list(
                session.scalars(
                    select(AnswerCitation).where(AnswerCitation.message_id == m.message_id)
                )
            )
            turn_payload.append(
                {
                    "turn": m.turn,
                    "role": m.role,
                    "content": m.content,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    "citations": [
                        {
                            "index": c.citation_index,
                            "chunk_id": str(c.chunk_id) if c.chunk_id else None,
                            "locator": c.locator,
                        }
                        for c in citations
                    ],
                }
            )
        out.append(
            {
                "conversation_id": str(conv.conversation_id),
                "created_at": conv.created_at.isoformat() if conv.created_at else None,
                "summary": conv.summary,
                "turns": turn_payload,
            }
        )
    return {
        "principal_id": principal_id,
        "conversations": out,
        "truncated": len(conversations) == max_conversations,
    }
