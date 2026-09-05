"""Current-state views and SQL rendering over append-only datasets."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from akl.lakehouse.engine import QueryEngine
from akl.lakehouse.io import LakehouseIO, Layer
from akl.lakehouse.schemas import DatasetSchema
from akl.lakehouse.schemas.gold import CHUNK_EMBEDDINGS, DEFAULT_EMBEDDING_VERSION, RETRIEVAL_UNITS
from akl.lakehouse.schemas.silver import CHUNKS, DOCUMENTS

SQL_DIR = Path(__file__).resolve().parent / "sql"
VIEWS_DIR = SQL_DIR / "views"
_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")
DatasetRef = tuple[Layer, str, DatasetSchema]
SourceResolver = Callable[[Layer, str, DatasetSchema], str]
DEFAULT_PARAMS: dict[str, str] = {
    "embedding_version": DEFAULT_EMBEDDING_VERSION,
    "chunker_version": "*",  # "*" = any; ChunkingService/GoldStore pass the active version (PRD §2.9)
    "chunk_config_hash": "*",
}


@dataclass(frozen=True)
class ViewDefinition:
    name: str
    sources: dict[str, DatasetRef]
    depends_on: tuple[str, ...] = ()
    params: tuple[str, ...] = field(default=())

    @property
    def sql_path(self) -> Path:
        return VIEWS_DIR / f"{self.name}.sql"


VIEWS: tuple[ViewDefinition, ...] = (
    ViewDefinition("v_current_documents", {"documents": (Layer.SILVER, "documents", DOCUMENTS)}),
    ViewDefinition(
        "v_current_chunks", {"chunks": (Layer.SILVER, "chunks", CHUNKS)}, ("v_current_documents",)
    ),
    ViewDefinition(
        "v_gold_active_units",
        {"retrieval_units": (Layer.GOLD, "retrieval_units", RETRIEVAL_UNITS)},
        ("v_current_chunks",),
    ),
    ViewDefinition(
        "v_embedding_coverage",
        {"chunk_embeddings": (Layer.GOLD, "chunk_embeddings", CHUNK_EMBEDDINGS)},
        ("v_gold_active_units",),
        ("embedding_version",),
    ),
)


def render_sql(
    template: str,
    *,
    sources: Mapping[str, DatasetRef],
    resolver: SourceResolver,
    params: Mapping[str, str] | None = None,
    context: str = "sql",
) -> str:
    params = params or {}

    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in params:
            return str(params[key]).replace("'", "''")
        if key in sources:
            layer, dataset, schema = sources[key]
            return resolver(layer, dataset, schema)
        raise KeyError(f"{context}: unknown placeholder {{{{ {key} }}}}")

    return _PLACEHOLDER.sub(substitute, template)


def render_view_sql(
    view: ViewDefinition, resolver: SourceResolver, params: Mapping[str, str] | None = None
) -> str:
    return render_sql(
        view.sql_path.read_text(encoding="utf-8"),
        sources=view.sources,
        resolver=resolver,
        params={**DEFAULT_PARAMS, **(params or {})},
        context=f"view {view.name}",
    )


class ViewRegistry:
    def __init__(
        self,
        engine: QueryEngine,
        io: LakehouseIO | None = None,
        resolver: SourceResolver | None = None,
        params: Mapping[str, str] | None = None,
    ) -> None:
        if io is None and resolver is None:
            raise ValueError("ViewRegistry needs a LakehouseIO or a custom resolver")
        self._engine = engine
        self._io = io
        self._resolver: SourceResolver = resolver or self._default_resolver
        self.params: dict[str, str] = {**DEFAULT_PARAMS, **(params or {})}

    @property
    def resolver(self) -> SourceResolver:
        return self._resolver

    def _default_resolver(self, layer: Layer, dataset: str, schema: DatasetSchema) -> str:
        assert self._io is not None
        if self._io.list_files(layer, dataset):
            return self._io.read_expression(layer, dataset)
        empty_name = f"_empty_{layer.value}_{dataset.replace('/', '_')}"
        self._engine.register(empty_name, schema.empty_table())
        return empty_name

    def register_all(self) -> list[str]:
        created: list[str] = []
        pending = list(VIEWS)
        while pending:
            progressed = False
            for view in list(pending):
                if all(dependency in created for dependency in view.depends_on):
                    self._engine.execute(
                        f"CREATE OR REPLACE VIEW {view.name} AS {render_view_sql(view, self._resolver, self.params)}"
                    )
                    created.append(view.name)
                    pending.remove(view)
                    progressed = True
            if not progressed:
                raise RuntimeError(f"view dependency cycle among {[view.name for view in pending]}")
        return created

    def counts(self) -> dict[str, int]:
        return {
            view.name: int(self._engine.execute_scalar(f"SELECT count(*) FROM {view.name}"))  # noqa: S608 - view names are registry constants
            for view in VIEWS
        }
