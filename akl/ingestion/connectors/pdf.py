"""PDF inbox connector (PRD §3.3.1): watches a directory / mounted volume for PDFs."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from akl.ingestion.connectors.base import ConnectorConfig
from akl.ingestion.connectors.directory import DirectoryConnector, DirectoryConnectorConfig
from akl.ingestion.models import SourceType


class PdfConnectorConfig(DirectoryConnectorConfig):
    type: Literal["pdf"] = "pdf"
    include_globs: list[str] = Field(default_factory=lambda: ["**/*.pdf", "**/*.PDF"])


class PdfConnector(DirectoryConnector):
    name = "pdf"
    version = "1.0.0"
    source_type: SourceType = "pdf"
    mime_type: ClassVar[str] = "application/pdf"
    config_cls: ClassVar[type[ConnectorConfig]] = PdfConnectorConfig
