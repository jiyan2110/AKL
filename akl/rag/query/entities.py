"""Entity extraction (PRD §6.2.4): error codes, env vars, paths, repos, versions, identifiers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

_PATTERNS: dict[str, re.Pattern[str]] = {
    "error_code": re.compile(r"\bAKL-[EW]\d{4}\b"),
    "env_var": re.compile(r"\b(?:AKL|AIRFLOW|MINIO|POSTGRES|QDRANT)_[A-Z0-9_]+\b"),
    "url": re.compile(r"https?://\S+"),
    "path": re.compile(r"\b[\w.-]+(?:/[\w.-]+)+\.[a-z0-9]{1,6}\b"),
    "version": re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b"),
    "identifier": re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b|\b[a-z]+(?:[A-Z][a-z0-9]+)+\b"),
    "backtick": re.compile(r"`([^`]+)`"),
}
_REPO = re.compile(r"\b([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)\b")
_QUOTED = re.compile(r"\"([^\"]{3,80})\"|'([^']{3,80})'")


@dataclass
class Entities:
    error_codes: list[str] = field(default_factory=list)
    env_vars: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    repos: list[str] = field(default_factory=list)
    versions: list[str] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    quoted: list[str] = field(default_factory=list)

    @property
    def exact_terms(self) -> list[str]:
        """Terms that must match exactly in sparse retrieval."""
        return list(
            dict.fromkeys(
                [*self.error_codes, *self.env_vars, *self.identifiers, *self.paths, *self.quoted]
            )
        )

    def as_dict(self) -> dict[str, list[str]]:
        return {k: v for k, v in self.__dict__.items() if v}


def extract_entities(text: str, *, known_repos: Iterable[str] = ()) -> Entities:
    ents = Entities()
    ents.error_codes = list(dict.fromkeys(_PATTERNS["error_code"].findall(text)))
    ents.env_vars = list(dict.fromkeys(_PATTERNS["env_var"].findall(text)))
    ents.urls = list(dict.fromkeys(_PATTERNS["url"].findall(text)))
    ents.paths = [
        p
        for p in dict.fromkeys(_PATTERNS["path"].findall(text))
        if not p.startswith(("http:", "https:"))
    ]
    ents.versions = list(dict.fromkeys(_PATTERNS["version"].findall(text)))
    ents.identifiers = [
        i for i in dict.fromkeys(_PATTERNS["identifier"].findall(text)) if i not in ents.env_vars
    ]
    ents.quoted = [a or b for a, b in _QUOTED.findall(text)]
    ents.quoted += [m for m in _PATTERNS["backtick"].findall(text) if m not in ents.quoted]
    repos_lower = {r.lower(): r for r in known_repos}
    for owner, name in _REPO.findall(text):
        candidate = f"{owner}/{name}"
        if candidate.lower() in repos_lower:
            ents.repos.append(repos_lower[candidate.lower()])
        elif candidate in ents.paths:
            continue
    ents.repos = list(dict.fromkeys(ents.repos))
    return ents
