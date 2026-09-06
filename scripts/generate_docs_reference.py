#!/usr/bin/env python3
"""Generate ``docs/reference/errors.md`` and ``docs/reference/metrics.md`` from the codebase
itself (PRD §11.11), so the docs site can never silently drift from what the code actually does.

Run via ``make docs-reference`` (also runs automatically before ``make docs-build``).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AKL_DIR = ROOT / "akl"
OUT_DIR = ROOT / "docs" / "reference"

_CODE_RE = re.compile(r"^AKL-[EW]\d{4}$")


def _class_docstring(node: ast.ClassDef) -> str:
    doc = ast.get_docstring(node) or ""
    return doc.strip().splitlines()[0].strip() if doc.strip() else ""


def _string_literal(value: ast.expr) -> str | None:
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def scan_error_codes() -> list[dict[str, str]]:
    """Every class with a ``code = "AKL-Exxxx"`` (or ``AKL-Wxxxx``) class attribute, across akl/."""
    found: dict[str, dict[str, str]] = {}
    for path in sorted(AKL_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                    continue
                target = stmt.targets[0]
                if not (isinstance(target, ast.Name) and target.id == "code"):
                    continue
                code = _string_literal(stmt.value)
                if code and _CODE_RE.match(code):
                    found[code] = {
                        "code": code,
                        "class": node.name,
                        "module": str(path.relative_to(ROOT)),
                        "description": _class_docstring(node) or node.name,
                    }
    return sorted(found.values(), key=lambda row: row["code"])


def render_errors_md(rows: list[dict[str, str]]) -> str:
    lines = [
        "# Error Code Reference",
        "",
        'Generated from `code = "AKL-..."` class attributes across the codebase by '
        "`scripts/generate_docs_reference.py` — do not edit by hand; regenerate with "
        "`make docs-reference`.",
        "",
        "| Code | Class | Description | Source |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['code']}` | `{row['class']}` | {row['description']} | `{row['module']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def render_metrics_md() -> str:
    sys.path.insert(0, str(ROOT))
    from prometheus_client.metrics import MetricWrapperBase

    from akl.observability import metrics as m

    rows: list[tuple[str, str, str, str]] = []
    for name in dir(m):
        obj = getattr(m, name)
        if isinstance(obj, MetricWrapperBase):
            kind = type(obj).__name__
            rows.append((obj._name, kind, obj._documentation, ", ".join(obj._labelnames)))
    rows.sort()
    lines = [
        "# Metrics Reference",
        "",
        "Generated from the module-level Prometheus metric objects in `akl/observability/metrics.py` "
        "(the scraped API registry) by `scripts/generate_docs_reference.py` — do not edit by hand; "
        "regenerate with `make docs-reference`. `PipelineMetrics` (pushed per Airflow/CLI task via "
        "Pushgateway, one fresh registry per task) is instantiated at runtime and is not listed here; "
        "see its class docstring in the same module for that catalog.",
        "",
        "| Metric | Type | Labels | Description |",
        "|---|---|---|---|",
    ]
    for metric_name, kind, doc, labels in rows:
        lines.append(f"| `{metric_name}` | {kind} | {labels or '—'} | {doc} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = scan_error_codes()
    (OUT_DIR / "errors.md").write_text(render_errors_md(rows), encoding="utf-8")
    (OUT_DIR / "metrics.md").write_text(render_metrics_md(), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'errors.md'} ({len(rows)} error codes)")
    print(f"wrote {OUT_DIR / 'metrics.md'}")


if __name__ == "__main__":
    main()
