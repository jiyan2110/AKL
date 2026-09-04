"""Document ingestion engine (PRD Chapter 3).

* :mod:`akl.ingestion.models` — SourceItem, FetchedObject, blocks, UnifiedDocument.
* :mod:`akl.ingestion.connectors` — BaseConnector, ConnectorRunner.
* :mod:`akl.ingestion.parsers` — BaseParser, DocumentAssembler.
* :mod:`akl.ingestion.registry` — connector/parser registries.
"""

from akl.ingestion.models import FetchedObject, SourceItem, UnifiedDocument

__all__ = ["FetchedObject", "SourceItem", "UnifiedDocument"]
