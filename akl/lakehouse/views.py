"""Current-state views over append-only Silver datasets."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from akl.lakehouse.engine import QueryEngine
from akl.lakehouse.io import LakehouseIO, Layer
from akl.lakehouse.schemas import DatasetSchema
from akl.lakehouse.schemas.silver import CHUNKS, DOCUMENTS

SQL_DIR = Path(__file__).resolve().parent / "sql" / "views"
_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")
DatasetRef = tuple[Layer, str, DatasetSchema]
SourceResolver = Callable[[Layer, str, DatasetSchema], str]


@dataclass(frozen=True)
class ViewDefinition:
    name: str
    sources: dict[str, DatasetRef]
    depends_on: tuple[str, ...] = ()

    @property
    def sql_path(self) -> Path:
        return SQL_DIR / f"{self.name}.sql"


VIEWS: tuple[ViewDefinition, ...] = (
    ViewDefinition(
        name="v_current_documents",
        sources={"documents": (Layer.SILVER, "documents", DOCUMENTS)},
    ),
    ViewDefinition(
        name="v_current_chunks",
        sources={"chunks": (Layer.SILVER, "chunks", CHUNKS)},
        depends_on=("v_current_documents",),
    ),
)


def render_view_sql(view: ViewDefinition, resolver: SourceResolver) -> str:
    template = view.sql_path.read_text(encoding="utf-8")

    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in view.sources:
            raise KeyError(f"view {view.name}: unknown placeholder {{{{ {key} }}}}")
        layer, dataset, schema = view.sources[key]
        return resolver(layer, dataset, schema)

    return _PLACEHOLDER.sub(substitute, template)


class ViewRegistry:
    def __init__(
        self,
        engine: QueryEngine,
        io: LakehouseIO | None = None,
        resolver: SourceResolver | None = None,
    ) -> None:
        if io is None and resolver is None:
            raise ValueError("ViewRegistry needs a LakehouseIO or a custom resolver")
        self._engine = engine
        self._io = io
        self._resolver = resolver or self._default_resolver

    def _default_resolver(self, layer: Layer, dataset: str, schema: DatasetSchema) -> str:
        assert self._io is not None
        if self._io.list_files(layer, dataset):
            return self._io.read_expression(layer, dataset)
        empty_name = f"_empty_{layer.value}_{dataset}"
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
                        f"CREATE OR REPLACE VIEW {view.name} AS {render_view_sql(view, self._resolver)}"
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
