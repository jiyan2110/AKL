"""Unit tests for the ingestion foundation (Milestone 10)."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pyarrow as pa
import pytest

from akl.ingestion.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorError,
    ConnectorRunner,
    PathRule,
)
from akl.ingestion.models import (
    CodeBlock,
    DeletionEvent,
    FetchedObject,
    HeadingBlock,
    SourceItem,
    TableBlock,
    UnifiedDocument,
)
from akl.ingestion.parsers.base import BaseParser, DocumentAssembler, UnsupportedFormatError
from akl.ingestion.registry import ConnectorRegistry, ParserRegistry
from akl.lakehouse.schemas import enforce
from akl.lakehouse.schemas.silver import DOCUMENTS
from akl.lakehouse.silver import SilverStore

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- fakes
class FakePut:
    def __init__(self, sha: str, dedup: bool) -> None:
        self.object_key = f"bronze/raw/source_type=markdown/sha256={sha}.md"
        self.content_sha256 = sha
        self.size_bytes = 3
        self.deduplicated = dedup


class FakeBronze:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.manifests: list[list[dict[str, Any]]] = []
        self.calls: list[str] = []

    def put_raw(
        self, data: bytes, *, source_type: str, mime_type: str | None, filename: str | None
    ) -> FakePut:
        sha = hashlib.sha256(data).hexdigest()
        self.calls.append("put")
        dedup = sha in self.objects
        self.objects[sha] = data
        return FakePut(sha, dedup)

    @staticmethod
    def build_manifest_row(**kw: Any) -> dict[str, Any]:
        return {"uri": kw["source_uri"], "sha": kw["put"].content_sha256, "run_id": kw["run_id"]}

    def write_manifest(self, rows: Sequence[Mapping[str, Any]], *, run_id: str) -> None:
        self.calls.append("manifest")
        self.manifests.append([dict(r) for r in rows])


class FakeDocs:
    def __init__(self, bronze: FakeBronze) -> None:
        self.records: list[dict[str, Any]] = []
        self.bronze = bronze

    def record_bronze(self, **kw: Any) -> None:
        self.bronze.calls.append("record")
        self.records.append(kw)


class FakeConnector(BaseConnector):
    name = "fake"
    version = "1.0.0"
    source_type = "markdown"
    retry_base_seconds = 0.0

    def __init__(
        self,
        config: ConnectorConfig,
        *,
        fail_first: set[str] | None = None,
        always_fail: set[str] | None = None,
    ) -> None:
        super().__init__(config)
        self.fail_first = set(fail_first or ())
        self.always_fail = set(always_fail or ())
        self.attempts: dict[str, int] = {}

    def discover(self, state: Mapping[str, Any]) -> Iterator[SourceItem | DeletionEvent]:
        seen = set(state.get("seen", []))
        for name in ("a", "b", "b-dup", "c"):
            if f"{name}.md" in seen:
                continue
            yield SourceItem(
                uri=f"file:///docs/{name}.md",
                canonical_uri=f"file:///docs/{name}.md",
                source_type="markdown",
                filename=f"{name}.md",
            )
        yield DeletionEvent(canonical_uri="file:///docs/gone.md")

    def fetch(self, item: SourceItem) -> FetchedObject:
        name = item.filename or ""
        self.attempts[name] = self.attempts.get(name, 0) + 1
        if name in self.always_fail or (name in self.fail_first and self.attempts[name] == 1):
            raise ConnectorError("boom", retryable=True)
        content = b"dup" if "dup" in name or name == "b.md" else name.encode()
        return FetchedObject.from_bytes(item, content, mime_type="text/markdown")

    def checkpoint(
        self, state: Mapping[str, Any], committed: Sequence[FetchedObject]
    ) -> dict[str, Any]:
        names = {c.item.filename for c in committed if c.item.filename}
        return {"seen": sorted(set(state.get("seen", [])) | names)}


def cfg(**over: Any) -> ConnectorConfig:
    return ConnectorConfig(id="fake-md", type="markdown", fetch_concurrency=2, **over)


# --------------------------------------------------------------------------- runner
def test_runner_full_flow_with_retry_dedup_and_deletion() -> None:
    bronze = FakeBronze()
    docs = FakeDocs(bronze)
    connector = FakeConnector(cfg(), fail_first={"a.md"})
    report = ConnectorRunner(bronze, docs).run(connector, state={}, run_id="r1")

    assert report.discovered == 4
    assert report.fetched == 4
    assert report.failed == 0
    assert report.deduplicated == 1  # b.md and b-dup.md share content
    assert len(bronze.objects) == 3
    assert report.manifest_rows == 4
    assert len(bronze.manifests) == 1
    assert [d.canonical_uri for d in report.deletions] == ["file:///docs/gone.md"]
    assert report.new_state == {"seen": ["a.md", "b-dup.md", "b.md", "c.md"]}
    assert connector.attempts["a.md"] == 2  # retried once
    assert bronze.calls[-1] == "manifest"  # manifest written last
    assert bronze.calls[0] == "put"  # a put always precedes any record
    assert len(docs.records) == 4
    assert all(r["connector_id"] == "fake-md" for r in docs.records)


def test_runner_isolates_failures_and_does_not_checkpoint_them() -> None:
    bronze = FakeBronze()
    connector = FakeConnector(cfg(), always_fail={"c.md"})
    connector.retry_attempts = 2
    report = ConnectorRunner(bronze).run(connector, state={"seen": ["a.md"]}, run_id="r2")
    assert report.discovered == 3
    assert report.fetched == 2
    assert report.failed == 1
    assert report.failures[0]["code"] == "AKL-E3020"
    assert "c.md" not in report.new_state["seen"]
    assert "a.md" in report.new_state["seen"]


def test_runner_respects_max_items() -> None:
    bronze = FakeBronze()
    report = ConnectorRunner(bronze).run(
        FakeConnector(cfg(max_items_per_run=2)), state={}, run_id="r3"
    )
    assert report.discovered == 4
    assert report.fetched == 2


def test_no_change_run_writes_nothing() -> None:
    bronze = FakeBronze()
    state = {"seen": ["a.md", "b.md", "b-dup.md", "c.md"]}
    report = ConnectorRunner(bronze).run(FakeConnector(cfg()), state=state, run_id="r4")
    assert report.fetched == 0
    assert report.manifest_rows == 0
    assert bronze.manifests == []
    assert report.new_state == {"seen": sorted(state["seen"])}


# --------------------------------------------------------------------------- config
def test_path_rules_raise_but_never_lower_security() -> None:
    c = ConnectorConfig(
        id="x1",
        type="github",
        security_level="internal",
        path_rules=[
            PathRule(glob="docs/public/*", security_level="public"),
            PathRule(glob="docs/hr/*", security_level="restricted", allowed_groups=["hr"]),
        ],
    )
    assert c.resolve_security("docs/public/faq.md") == ("internal", ())
    assert c.resolve_security("docs/hr/pay.md") == ("restricted", ("hr",))
    assert c.resolve_security("README.md") == ("internal", ())


# --------------------------------------------------------------------------- assembler
def test_assembler_offsets_headings_pages_and_silver_row() -> None:
    a = DocumentAssembler()
    a.add_heading(1, "Install")
    a.add_paragraph("First para.\r\n\r\n\r\n\r\nWith  gaps.")
    a.add_heading(2, "Docker")
    a.add_code("x = 1\n", language="python")
    a.page_break()
    a.add_table("| a | b |\n|---|---|\n| 1 | 2 |", n_rows=1, n_cols=2, caption="Ports")
    a.add_heading(1, "Usage")
    doc = a.build(
        source_uri="https://Ex.com/Doc/",
        source_type="html",
        content_sha256="a" * 64,
        parser_name="t",
        parser_version="1.0.0",
    )

    for b in doc.blocks:
        if isinstance(b, HeadingBlock):
            assert doc.text[b.start_char : b.end_char] == f"{'#' * b.level} {b.text}"
    code = next(b for b in doc.blocks if isinstance(b, CodeBlock))
    assert doc.text[code.start_char : code.end_char] == "```python\nx = 1\n```"
    table = next(b for b in doc.blocks if isinstance(b, TableBlock))
    assert doc.text[table.start_char : table.end_char].startswith("Ports")

    assert doc.title == "Install"
    assert [n.text for n in doc.structure] == ["Install", "Usage"]
    assert [c.text for c in doc.structure[0].children] == ["Docker"]
    usage = next(b for b in doc.blocks if isinstance(b, HeadingBlock) and b.text == "Usage")
    assert doc.structure[0].end_char == usage.start_char
    assert [p["page"] for p in doc.page_map] == [1, 2]
    assert doc.canonical_source_uri == "https://ex.com/Doc"
    assert "\r" not in doc.text
    assert "\n\n\n" not in doc.text
    assert isinstance(doc, UnifiedDocument)

    row = SilverStore.prepare_document_row(doc.to_silver_row())
    table_out = enforce(pa.Table.from_pylist([row], schema=DOCUMENTS.schema), DOCUMENTS)
    assert table_out.num_rows == 1


# --------------------------------------------------------------------------- registries
class MdParser(BaseParser):
    name = "md"
    version = "1"
    mime_types = ("text/markdown",)
    extensions = ("md",)

    def parse(self, obj: FetchedObject) -> UnifiedDocument:
        raise NotImplementedError


def test_parser_registry_selects_first_match_or_raises() -> None:
    reg = ParserRegistry()
    reg.register(MdParser())
    assert reg.select("text/markdown; charset=utf-8", "", "github").name == "md"
    assert reg.select(None, "MD", "markdown").name == "md"
    with pytest.raises(UnsupportedFormatError):
        reg.select("application/zip", "zip", "pdf")


def test_connector_registry_creates_by_type() -> None:
    reg = ConnectorRegistry()
    reg.register(FakeConnector)
    assert "markdown" in reg
    conn = reg.create(cfg())
    assert isinstance(conn, FakeConnector)
    assert conn.config.id == "fake-md"
    with pytest.raises(KeyError):
        reg.connector_class("pdf")
