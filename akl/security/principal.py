"""Principal model and the document-access predicate (PRD §9.3).

A principal may read a chunk iff ``chunk.security_level ∈ principal.security_levels``
and (``chunk.allowed_groups`` is empty or intersects ``principal.groups``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

SECURITY_ORDER = {"public": 0, "internal": 1, "restricted": 2}


@dataclass(frozen=True)
class Principal:
    subject: str
    groups: frozenset[str] = field(default_factory=frozenset)
    security_levels: frozenset[str] = field(default_factory=lambda: frozenset({"public"}))
    scopes: frozenset[str] = field(default_factory=lambda: frozenset({"search:read", "chat:write"}))

    @classmethod
    def dev(cls) -> Principal:
        """Local-development principal: every level, no group restriction bypass."""
        return cls("dev", frozenset({"eng"}), frozenset(SECURITY_ORDER), frozenset({"*"}))

    def can_read(
        self, security_level: str, allowed_groups: list[str] | tuple[str, ...] | None
    ) -> bool:
        if security_level not in self.security_levels:
            return False
        return not allowed_groups or bool(self.groups & set(allowed_groups))

    def has_scope(self, scope: str) -> bool:
        return "*" in self.scopes or scope in self.scopes
