"""Unit tests: GitHub connector and text/code parsers (Milestone 14)."""

from __future__ import annotations

import base64
from typing import Any

import httpx
import pytest

from akl.ingestion.connectors.github import (
    GitHubConnector,
    GitHubConnectorConfig,
    GitHubNotFoundError,
)
from akl.ingestion.models import CodeBlock, DeletionEvent, FetchedObject, HeadingBlock, SourceItem
from akl.ingestion.parsers.text import CodeParser, TextParser

pytestmark = pytest.mark.unit


class FakeGitHub:
    def __init__(self) -> None:
        self.head = "c1"
        self.tree: dict[str, tuple[str, int]] = {
            "README.md": ("s1", 10),
            "docs/a.md": ("s2", 20),
            "src/x.py": ("s3", 30),
            "CHANGELOG.md": ("s4", 5),
            "big.md": ("s5", 5_000_000),
        }
        self.remaining = 4999
        self.blob_calls = 0

    def handler(self, req: httpx.Request) -> httpx.Response:
        p = req.url.path
        h = {"x-ratelimit-remaining": str(self.remaining)}
        if p.endswith("/branches/main"):
            return httpx.Response(200, json={"commit": {"sha": self.head}}, headers=h)
        if "/git/trees/" in p:
            entries: list[dict[str, Any]] = [
                {"path": k, "type": "blob", "sha": v[0], "size": v[1]} for k, v in self.tree.items()
            ]
            return httpx.Response(200, json={"truncated": False, "tree": entries}, headers=h)
        if "/git/blobs/" in p:
            self.blob_calls += 1
            sha = p.rsplit("/", 1)[1]
            return httpx.Response(
                200,
                json={
                    "encoding": "base64",
                    "content": base64.b64encode(f"content-{sha}".encode()).decode(),
                },
                headers=h,
            )
        return httpx.Response(404, headers=h)


def _conn(fake: FakeGitHub, **over: Any) -> GitHubConnector:
    cfg = GitHubConnectorConfig(
        id="gh", type="github", owner="Org", repo="Repo", include_code=True, mode="api", **over
    )
    return GitHubConnector(cfg, transport=httpx.MockTransport(fake.handler))


def test_github_initial_run_diff_and_deletion() -> None:
    fake = FakeGitHub()
    conn = _conn(fake)
    events = list(conn.discover({}))
    items = [e for e in events if isinstance(e, SourceItem)]
    assert [i.source_metadata["git.path"] for i in items] == [
        "README.md",
        "docs/a.md",
        "src/x.py",
    ]  # CHANGELOG excluded, big.md over size
    assert items[0].canonical_uri == "github://org/Repo/main/README.md"
    assert items[0].uri == "github://Org/Repo/c1/README.md"
    fetched = [conn.fetch(i) for i in items]
    assert fetched[0].data == b"content-s1"
    assert fetched[2].mime_type == "text/plain"
    state = conn.checkpoint({}, fetched)
    assert state["last_commit_sha"] == "c1"
    assert len(conn.snapshot_rows) == 3

    # unchanged head → zero items, zero blob calls
    fake.blob_calls = 0
    conn2 = _conn(fake)
    assert list(conn2.discover(state)) == []
    assert fake.blob_calls == 0

    # new commit: README modified, docs/a.md deleted
    fake.head = "c2"
    fake.tree = {"README.md": ("s1b", 10), "src/x.py": ("s3", 30)}
    conn3 = _conn(fake)
    events3 = list(conn3.discover(state))
    assert [e.source_metadata["git.path"] for e in events3 if isinstance(e, SourceItem)] == [
        "README.md"
    ]
    assert [e.canonical_uri for e in events3 if isinstance(e, DeletionEvent)] == [
        "github://org/Repo/main/docs/a.md"
    ]
    state3 = conn3.checkpoint(state, [conn3.fetch(e) for e in events3 if isinstance(e, SourceItem)])
    assert state3["last_commit_sha"] == "c2"
    assert sorted(state3["tree"]) == ["README.md", "src/x.py"]


def test_github_partial_checkpoint_does_not_advance_commit() -> None:
    fake = FakeGitHub()
    conn = _conn(fake)
    items = [e for e in conn.discover({}) if isinstance(e, SourceItem)]
    state = conn.checkpoint({}, [conn.fetch(items[0])])  # only README committed
    assert "last_commit_sha" not in state
    assert list(state["tree"]) == ["README.md"]


def test_github_not_found() -> None:
    fake = FakeGitHub()
    conn = GitHubConnector(
        GitHubConnectorConfig(
            id="gh", type="github", owner="o", repo="r", branch="nope", mode="api"
        ),
        transport=httpx.MockTransport(fake.handler),
    )
    with pytest.raises(GitHubNotFoundError):
        list(conn.discover({}))
    assert conn.health().ok is False


def _fetched(text: str, name: str) -> FetchedObject:
    item = SourceItem(
        uri=f"github://o/r/main/{name}",
        canonical_uri=f"github://o/r/main/{name}",
        source_type="github",
        filename=name,
        source_metadata={"git.path": name},
    )
    return FetchedObject.from_bytes(item, text.encode(), mime_type=None)


def test_text_parser_rst_headings_and_paragraphs() -> None:
    rst = "Title\n=====\n\nFirst para line one\nline two.\n\nSub\n---\n\n# atx heading\n\nSecond para.\n"
    doc = TextParser().parse(_fetched(rst, "README.rst"))
    heads = [b for b in doc.blocks if isinstance(b, HeadingBlock)]
    assert [(h.level, h.text) for h in heads] == [(1, "Title"), (2, "Sub"), (1, "atx heading")]
    assert doc.blocks[1].text == "First para line one line two."  # type: ignore[union-attr]


def test_code_parser_wraps_file() -> None:
    doc = CodeParser().parse(_fetched("def f():\n    return 1\n", "src/x.py"))
    code = next(b for b in doc.blocks if isinstance(b, CodeBlock))
    assert code.language == "python"
    assert doc.title == "src/x.py"
    assert doc.metadata["code.language"] == "python"
