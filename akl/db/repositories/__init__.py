"""Repositories: all SQL against the ``akl`` database lives here (PRD §15.2).

Services receive a repository bound to a session; they never build SQL themselves.
"""

from __future__ import annotations

from sqlalchemy.orm import Session


class Repository:
    """Base class holding the SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        self.session = session


__all__ = ["Repository"]
