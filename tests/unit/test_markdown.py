"""Unit tests: Markdown connector + parser (Milestone 11)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.test_ingestion_base import FakeBronze

from akl.ingestion.connectors.base import ConnectorRunner
from akl.ingestion.connectors.directory import glob_match
from akl.ingestion.connectors.markdown import MarkdownConnector, MarkdownConnectorConfig
from akl.ingestion.models import (
    CodeBlock,
    DeletionEvent,
    FetchedObject,
    HeadingBlock,
    ListBlock,
    SourceItem,
    TableBlock,
)
from akl.ingestion.parsers.markdown import MarkdownParser, strip_mdx

pytestmark = pytest.mark.unit

SAMPLE = """---
title: Deploying AKL
tags: [ops, docker]
security_level: restricted
---

# Overview

AKL runs on **Docker Compose** with `make up`. See [the PRD](PRD.md).

## Steps

1. Copy `.env.example`
2. Run `make up`

| Service | Port |
|---------|------|
| API     | 8000 |
| MinIO   | 9000 |

```bash
make up && make wait
```

<div class="note">Rendered <b>HTML</b> block</div>
"""


def fetched(text: str, name: str = "deploy.md", level: str = "internal") -> FetchedObject:
    item = SourceItem(
        uri=f"https://docs.example.com/{name}",
        canonical_uri=f"https://docs.example.com/{name}",
        source_type="markdown",
        filename=name,
        security_level=level,  # type: ignore[arg-type]
    )
    return FetchedObject.from_bytes(item, text.encode(), mime_type="text/markdown")


def test_parser_frontmatter_title_and_security_raise_only() -> None:
    doc = MarkdownParser().parse(fetched(SAMPLE))
    assert doc.title == "Deploying AKL"
    assert doc.security_level == "restricted"
    assert doc.metadata["frontmatter.tags"] == "ops, docker"
    lowered = MarkdownParser().parse(
        fetched(SAMPLE.replace("restricted", "public"), level="internal")
    )
    assert lowered.security_level == "internal"  # frontmatter may not lower the level


def test_parser_blocks_and_offsets() -> None:
    doc = MarkdownParser().parse(fetched(SAMPLE))
    kinds = [b.kind for b in doc.blocks]
    assert kinds == ["heading", "paragraph", "heading", "list", "table", "code", "paragraph"]
    para = doc.blocks[1]
    assert (
        doc.text[para.start_char : para.end_char]
        == "AKL runs on Docker Compose with `make up`. See [the PRD](PRD.md)."
    )
    lst = next(b for b in doc.blocks if isinstance(b, ListBlock))
    assert lst.ordered is True
    assert lst.items == ("Copy `.env.example`", "Run `make up`")
    table = next(b for b in doc.blocks if isinstance(b, TableBlock))
    assert table.n_rows == 2
    assert table.n_cols == 2
    assert table.markdown.splitlines()[0] == "| Service | Port |"
    code = next(b for b in doc.blocks if isinstance(b, CodeBlock))
    assert code.language == "bash"
    assert doc.text[code.start_char : code.end_char] == "```bash\nmake up && make wait\n```"
    assert doc.blocks[-1].text == "Rendered HTML block"  # type: ignore[union-attr]
    assert [h.text for h in doc.structure] == ["Overview"]
    assert [c.text for c in doc.structure[0].children] == ["Steps"]


def test_mdx_stripping() -> None:
    mdx = 'import Tabs from "@theme/Tabs"\n\n# Title\n\n<Tabs>\n<TabItem value="a">Body text here.</TabItem>\n</Tabs>\n'
    assert "import" not in strip_mdx(mdx)
    doc = MarkdownParser().parse(fetched(mdx, name="page.mdx"))
    assert [b.kind for b in doc.blocks] == ["heading", "paragraph"]
    assert isinstance(doc.blocks[0], HeadingBlock)


def test_parser_supports() -> None:
    p = MarkdownParser()
    assert p.supports("text/markdown; charset=utf-8", "", "markdown")
    assert p.supports(None, "MDX", "github")
    assert not p.supports("application/pdf", "pdf", "pdf")


@pytest.mark.parametrize(
    ("rel", "pattern", "expected"),
    [
        ("a.md", "**/*.md", True),
        ("x/y/a.md", "**/*.md", True),
        ("node_modules/a.md", "**/node_modules/**", True),
        ("a/node_modules/b.md", "**/node_modules/**", True),
        ("a.txt", "**/*.md", False),
    ],
)
def test_glob_match(rel: str, pattern: str, expected: bool) -> None:
    assert glob_match(rel, pattern) is expected


def _cfg(root: Path, **over: object) -> MarkdownConnectorConfig:
    return MarkdownConnectorConfig(
        id="md-test",
        type="markdown",
        root_path=root,
        uri_base="https://docs.example.com/x",
        fetch_concurrency=2,
        **over,
    )  # type: ignore[arg-type]


def test_connector_discover_fetch_checkpoint_and_deletion(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\ntext a\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("# B\n\ntext b\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    (tmp_path / "drafts").mkdir()
    (tmp_path / "drafts" / "d.md").write_text("draft", encoding="utf-8")
    conn = MarkdownConnector(_cfg(tmp_path, exclude_globs=["**/drafts/**"]))

    report = ConnectorRunner(FakeBronze()).run(conn, state={}, run_id="r1")
    assert report.discovered == 2
    assert report.fetched == 2
    assert sorted(report.new_state["files"]) == ["a.md", "sub/b.md"]
    uris = sorted(c.item.uri for c in report.committed)
    assert uris == ["https://docs.example.com/x/a.md", "https://docs.example.com/x/sub/b.md"]

    # no change → nothing discovered
    conn2 = MarkdownConnector(_cfg(tmp_path, exclude_globs=["**/drafts/**"]))
    report2 = ConnectorRunner(FakeBronze()).run(conn2, state=report.new_state, run_id="r2")
    assert report2.discovered == 0

    # modify one, delete one
    (tmp_path / "a.md").write_text("# A changed\n\nmore text\n", encoding="utf-8")
    import os

    os.utime(tmp_path / "a.md", (2_000_000_000, 2_000_000_000))
    (tmp_path / "sub" / "b.md").unlink()
    conn3 = MarkdownConnector(_cfg(tmp_path, exclude_globs=["**/drafts/**"]))
    events = list(conn3.discover(report.new_state))
    assert [e.filename for e in events if isinstance(e, SourceItem)] == ["a.md"]
    assert [e.canonical_uri for e in events if isinstance(e, DeletionEvent)] == [
        "https://docs.example.com/x/sub/b.md"
    ]
    fetched_objs = [conn3.fetch(e) for e in events if isinstance(e, SourceItem)]
    state3 = conn3.checkpoint(report.new_state, fetched_objs)
    assert sorted(state3["files"]) == ["a.md"]


def test_connector_path_rules_apply_security(tmp_path: Path) -> None:
    (tmp_path / "public").mkdir()
    (tmp_path / "public" / "faq.md").write_text("# FAQ\n\nq&a\n", encoding="utf-8")
    (tmp_path / "hr.md").write_text("# HR\n\nsecret\n", encoding="utf-8")
    from akl.ingestion.connectors.base import PathRule

    cfg = _cfg(
        tmp_path,
        security_level="internal",
        path_rules=[
            PathRule(glob="public/*", security_level="public"),
            PathRule(glob="hr.md", security_level="restricted", allowed_groups=["hr"]),
        ],
    )
    items = {
        e.filename: e for e in MarkdownConnector(cfg).discover({}) if isinstance(e, SourceItem)
    }
    assert items["faq.md"].security_level == "internal"  # rules never lower
    assert items["hr.md"].security_level == "restricted"
    assert items["hr.md"].allowed_groups == ("hr",)
