"""Dataset schema registry and enforcement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from akl.errors import AKLError


class SchemaEnforcementError(AKLError):
    code = "AKL-E2101"
    http_status = 500
    retryable = False


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    version: str
    schema: pa.Schema
    partition_by: tuple[str, ...] = ()
    sort_by: tuple[str, ...] = ()
    description: str = field(default="", compare=False)

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.schema.names)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "partition_by": list(self.partition_by),
            "sort_by": list(self.sort_by),
            "fields": [
                {"name": field.name, "type": str(field.type), "nullable": field.nullable}
                for field in self.schema
            ],
        }

    def empty_table(self) -> pa.Table:
        return self.schema.empty_table()


def enforce(table: pa.Table, dataset: DatasetSchema) -> pa.Table:
    expected = dataset.schema
    unknown = [column for column in table.column_names if column not in expected.names]
    if unknown:
        raise SchemaEnforcementError(
            f"unknown columns for dataset '{dataset.name}'",
            details={"dataset": dataset.name, "version": dataset.version, "unknown": unknown},
        )
    arrays: list[pa.Array | pa.ChunkedArray] = []
    for schema_field in expected:
        if schema_field.name in table.column_names:
            arrays.append(table.column(schema_field.name))
        elif schema_field.nullable:
            arrays.append(pa.nulls(table.num_rows, type=schema_field.type))
        else:
            raise SchemaEnforcementError(
                f"missing required column '{schema_field.name}' for dataset '{dataset.name}'",
                details={
                    "dataset": dataset.name,
                    "version": dataset.version,
                    "missing": schema_field.name,
                },
            )
    ordered = pa.Table.from_arrays(arrays, names=list(expected.names))
    try:
        return ordered.cast(expected, safe=True)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError) as exc:
        raise SchemaEnforcementError(
            f"type cast failed for dataset '{dataset.name}'",
            details={"dataset": dataset.name, "version": dataset.version, "error": str(exc)},
        ) from exc
