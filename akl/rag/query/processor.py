"""QueryProcessor: the full PRD §6.2 pipeline producing a :class:`ProcessedQuery`."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from akl.config import RetrievalSettings
from akl.rag.query.entities import Entities, extract_entities
from akl.rag.query.filters import MetadataFilters, infer_soft_filters
from akl.rag.query.intent import INTENT_PROFILES, Intent, classify_intent
from akl.rag.query.normalize import NormalizedQuery, normalize_query
from akl.rag.query.spell import SpellCorrector
from akl.security.principal import Principal


@dataclass
class ProcessedQuery:
    original: str
    dense_text: str  # original casing, spell-corrected when dual mode is off
    sparse_text: str  # lower-cased, spell-corrected
    dense_variants: tuple[str, ...]  # texts to embed (original [+ corrected] when dual mode)
    tokens: tuple[str, ...]
    protected: frozenset[str]
    corrections: dict[str, str]
    intent: Intent
    entities: Entities
    hard_filters: MetadataFilters
    soft_filters: MetadataFilters
    principal: Principal
    profile: dict[str, Any] = field(default_factory=dict)

    @property
    def exact_terms(self) -> list[str]:
        return self.entities.exact_terms

    def trace(self) -> dict[str, Any]:
        return {
            "normalized": self.dense_text,
            "corrected": self.sparse_text,
            "corrections": self.corrections,
            "intent": self.intent.value,
            "entities": self.entities.as_dict(),
            "filters_hard": self.hard_filters.as_dict(),
            "filters_soft": self.soft_filters.as_dict(),
            "principal": {
                "subject": self.principal.subject,
                "levels": sorted(self.principal.security_levels),
            },
        }


class QueryProcessor:
    def __init__(
        self,
        settings: RetrievalSettings,
        *,
        spell: SpellCorrector | None = None,
        known_repos: Iterable[str] = (),
    ) -> None:
        self.settings = settings
        self.spell = spell
        self.known_repos = list(known_repos)

    def process(
        self, raw: str, principal: Principal, *, filters: MetadataFilters | None = None
    ) -> ProcessedQuery:
        norm: NormalizedQuery = normalize_query(raw, max_chars=self.settings.query_max_chars)
        corrected, corrections = (
            self.spell.correct(norm.text, norm.protected) if self.spell else (norm.text, {})
        )
        intent = classify_intent(norm.text)
        entities = extract_entities(norm.text, known_repos=self.known_repos)
        soft = infer_soft_filters(norm.text, entities)
        hard = filters or MetadataFilters()
        # user-supplied hard filters override inferred soft ones on the same dimension
        for dim in ("source_types", "repos", "chunk_types", "code_languages"):
            if getattr(hard, dim):
                setattr(soft, dim, [])
        if hard.updated_after:
            soft.updated_after = None
        dual = self.settings.query_spell_dual and corrections
        variants = (
            (norm.text, corrected) if dual else ((corrected,) if corrections else (norm.text,))
        )
        return ProcessedQuery(
            original=raw,
            dense_text=norm.text if dual else corrected,
            sparse_text=corrected.lower(),
            dense_variants=tuple(dict.fromkeys(variants)),
            tokens=norm.tokens,
            protected=norm.protected,
            corrections=corrections,
            intent=intent,
            entities=entities,
            hard_filters=hard,
            soft_filters=soft,
            principal=principal,
            profile=dict(INTENT_PROFILES[intent]),
        )
