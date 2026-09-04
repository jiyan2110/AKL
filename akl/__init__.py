"""Enterprise AI Knowledge Lakehouse (AKL).

Top-level package for the lakehouse, ingestion, chunking, embedding, retrieval,
API and orchestration modules. See ``PRD.md`` for the authoritative design.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__: str = version("akl")
except PackageNotFoundError:  # pragma: no cover - only when running from a raw checkout
    __version__ = "0.0.0+unknown"
