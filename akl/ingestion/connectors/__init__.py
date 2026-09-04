"""Source connectors (PRD §3.3). Each connector implements discover/fetch/checkpoint/health."""

from akl.ingestion.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorRunner,
    IngestionRunReport,
    PathRule,
)

__all__ = ["BaseConnector", "ConnectorConfig", "ConnectorRunner", "IngestionRunReport", "PathRule"]
