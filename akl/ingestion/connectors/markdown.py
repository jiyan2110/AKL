"""Markdown directory connector (PRD §3.3.2)."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from akl.ingestion.connectors.base import ConnectorConfig
from akl.ingestion.connectors.directory import DirectoryConnector, DirectoryConnectorConfig
from akl.ingestion.models import SourceType


class MarkdownConnectorConfig(DirectoryConnectorConfig):
    type: Literal["markdown"] = "markdown"
    include_globs: list[str] = Field(
        default_factory=lambda: ["**/*.md", "**/*.mdx", "**/*.markdown"]
    )


class MarkdownConnector(DirectoryConnector):
    name = "markdown"
    version = "1.0.0"
    source_type: SourceType = "markdown"
    mime_type: ClassVar[str] = "text/markdown"
    config_cls: ClassVar[type[ConnectorConfig]] = MarkdownConnectorConfig
