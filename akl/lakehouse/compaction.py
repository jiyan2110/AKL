"""Small-file compaction for append-only Parquet partitions (PRD §2.8).

Per partition: when the file count exceeds ``min_files`` or any file is smaller than
``small_file_mb``, all files are read (DuckDB), rewritten as one sorted file into the same
partition directory, row counts verified, and the originals deleted. Readers deduplicate by
primary key through the current-state views, so the brief overlap window is harmless.
``bronze/raw`` is never touched (immutable). Dry-run mode returns the plan only.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from akl.errors import AKLError
from akl.lakehouse.engine import QueryEngine
from akl.lakehouse.io import FileInfo, LakehouseIO, Layer

COMPACTABLE: tuple[tuple[Layer, str, tuple[str, ...]], ...] = (
    (Layer.BRONZE, "manifest", ("document_id", "fetched_at")),
    (Layer.SILVER, "documents", ("document_id", "parsed_at")),
    (Layer.SILVER, "chunks", ("document_id", "chunk_index")),
    (Layer.GOLD, "retrieval_units", ("document_id", "chunk_index")),
    (Layer.GOLD, "chunk_embeddings", ("chunk_id",)),
)


class CompactionError(AKLError):
    code = "AKL-E2102"
    retryable = False


@dataclass
class PartitionPlan:
    layer: str
    dataset: str
    partition: str  # "" for unpartitioned datasets
    files: list[str]
    bytes_before: int
    reason: str


@dataclass
class CompactionReport:
    planned: list[PartitionPlan] = field(default_factory=list)
    compacted: int = 0
    files_merged: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


def group_by_partition(dataset_prefix: str, files: list[FileInfo]) -> dict[str, list[FileInfo]]:
    groups: dict[str, list[FileInfo]] = defaultdict(list)
    for f in files:
        rel = f.key[len(dataset_prefix) :]
        partition = rel.rsplit("/", 1)[0] if "/" in rel else ""
        groups[partition].append(f)
    return dict(groups)


def plan_partitions(io: LakehouseIO, *, min_files: int, small_file_mb: int) -> list[PartitionPlan]:
    plans: list[PartitionPlan] = []
    small = small_file_mb * 1024 * 1024
    for layer, dataset, _sort in COMPACTABLE:
        files = io.list_files(layer, dataset)
        prefix = f"{layer.value}/{dataset}/"
        for partition, group in group_by_partition(prefix, files).items():
            if len(group) < 2:
                continue
            reasons = []
            if len(group) > min_files:
                reasons.append(f"{len(group)} files > {min_files}")
            if any(f.size_bytes < small for f in group):
                reasons.append(
                    f"{sum(1 for f in group if f.size_bytes < small)} files < {small_file_mb} MiB"
                )
            if reasons:
                plans.append(
                    PartitionPlan(
                        layer.value,
                        dataset,
                        partition,
                        [f.key for f in group],
                        sum(f.size_bytes for f in group),
                        "; ".join(reasons),
                    )
                )
    return plans


def compact(
    io: LakehouseIO,
    engine: QueryEngine,
    *,
    run_id: str,
    min_files: int = 8,
    small_file_mb: int = 32,
    dry_run: bool = False,
) -> CompactionReport:
    report = CompactionReport(
        planned=plan_partitions(io, min_files=min_files, small_file_mb=small_file_mb)
    )
    if dry_run:
        return report
    sort_keys = {(layer.value, ds): keys for layer, ds, keys in COMPACTABLE}
    for plan in report.planned:
        try:
            files_sql = ", ".join(f"'s3://{io.bucket}/{k}'" for k in plan.files)
            order = ", ".join(f'"{c}"' for c in sort_keys[(plan.layer, plan.dataset)])
            table = engine.execute(
                f"SELECT * FROM read_parquet([{files_sql}], union_by_name = true) ORDER BY {order}"
            )
            expected = int(
                engine.execute_scalar(
                    f"SELECT count(*) FROM read_parquet([{files_sql}], union_by_name = true)"
                )
                or 0
            )
            if table.num_rows != expected:
                raise CompactionError(
                    "row count mismatch before write",
                    details={
                        "partition": plan.partition,
                        "expected": expected,
                        "got": table.num_rows,
                    },
                )
            # partition columns come back as hive columns; drop them so the file layout stays clean
            partition_cols = [
                seg.split("=", 1)[0] for seg in plan.partition.split("/") if "=" in seg
            ]
            table = table.drop_columns([c for c in partition_cols if c in table.column_names])
            new_key = io.write_file(
                table,
                plan.layer,
                plan.dataset,
                partition=plan.partition,
                run_id=f"compact-{run_id}",
            )
            written = int(
                engine.execute_scalar(
                    f"SELECT count(*) FROM read_parquet('s3://{io.bucket}/{new_key}')"
                )
                or 0
            )
            if written != expected:
                io.delete_keys([new_key])
                raise CompactionError(
                    "row count mismatch after write",
                    details={"partition": plan.partition, "expected": expected, "got": written},
                )
            io.delete_keys(plan.files)
            after = next(
                (
                    f.size_bytes
                    for f in io.list_files(
                        plan.layer, plan.dataset, partition=plan.partition or None
                    )
                    if f.key == new_key
                ),
                0,
            )
            report.compacted += 1
            report.files_merged += len(plan.files)
            report.bytes_before += plan.bytes_before
            report.bytes_after += after
        except AKLError as exc:
            report.errors.append(f"{plan.layer}/{plan.dataset}/{plan.partition}: {exc.message}")
            report.skipped += 1
    return report
