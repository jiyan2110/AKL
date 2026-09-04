"""Connector and parser registries (PRD §3.2)."""

from __future__ import annotations

from importlib.metadata import entry_points

from akl.ingestion.connectors.base import BaseConnector, ConnectorConfig
from akl.ingestion.parsers.base import BaseParser, UnsupportedFormatError

CONNECTOR_ENTRY_POINT_GROUP = "akl.connectors"
PARSER_ENTRY_POINT_GROUP = "akl.parsers"


class ConnectorRegistry:
    def __init__(self) -> None:
        self._connectors: dict[str, type[BaseConnector]] = {}
        self._configs: dict[str, type[ConnectorConfig]] = {}

    def register(
        self,
        connector_cls: type[BaseConnector],
        config_cls: type[ConnectorConfig] = ConnectorConfig,
    ) -> None:
        self._connectors[connector_cls.source_type] = connector_cls
        self._configs[connector_cls.source_type] = config_cls

    def connector_class(self, source_type: str) -> type[BaseConnector]:
        try:
            return self._connectors[source_type]
        except KeyError as exc:
            raise KeyError(f"no connector registered for source_type {source_type!r}") from exc

    @property
    def config_types(self) -> dict[str, type[ConnectorConfig]]:
        return dict(self._configs)

    def create(self, config: ConnectorConfig) -> BaseConnector:
        return self.connector_class(config.type)(config)

    def load_entry_points(self) -> int:
        count = 0
        for ep in entry_points(group=CONNECTOR_ENTRY_POINT_GROUP):
            cls = ep.load()
            if isinstance(cls, type) and issubclass(cls, BaseConnector):
                self.register(cls, getattr(cls, "config_cls", ConnectorConfig))
                count += 1
        return count

    def __contains__(self, source_type: object) -> bool:
        return source_type in self._connectors


class ParserRegistry:
    def __init__(self) -> None:
        self._parsers: list[BaseParser] = []

    def register(self, parser: BaseParser) -> None:
        self._parsers.append(parser)

    def select(self, mime_type: str | None, extension: str, source_type: str) -> BaseParser:
        for parser in self._parsers:  # ordered: first match wins (PRD §3.4.1)
            if parser.supports(mime_type, extension, source_type):
                return parser
        raise UnsupportedFormatError(
            "no parser for input",
            details={"mime_type": mime_type, "extension": extension, "source_type": source_type},
        )

    def load_entry_points(self) -> int:
        count = 0
        for ep in entry_points(group=PARSER_ENTRY_POINT_GROUP):
            cls = ep.load()
            if isinstance(cls, type) and issubclass(cls, BaseParser):
                self.register(cls())
                count += 1
        return count

    def __len__(self) -> int:
        return len(self._parsers)


connectors = ConnectorRegistry()
parsers = ParserRegistry()
