"""Unit tests: HTML parser and connector (Milestone 13)."""

from __future__ import annotations

import httpx
import pytest

from akl.ingestion.connectors.html import HtmlConnector, HtmlConnectorConfig
from akl.ingestion.models import (
    CodeBlock,
    DeletionEvent,
    FetchedObject,
    ListBlock,
    SourceItem,
    TableBlock,
)
from akl.ingestion.parsers.html import HtmlParser

pytestmark = pytest.mark.unit

PAGE = """<!doctype html><html><head><title>Runbook: Restore</title>
<link rel="canonical" href="https://wiki.example.com/runbooks/restore">
<meta name="description" content="How to restore"></head><body>
<nav><a href="/">Home</a><a href="/x">Other</a></nav>
<main><h1>Restore Procedure</h1><p>Restore <b>MinIO</b> first, then <code>pg_restore</code>. See <a href="https://x.y/z">docs</a>.</p>
<h2>Steps</h2><ol><li>Stop API</li><li>Restore volume</li></ol>
<pre><code class="language-bash">akl-cli qdrant sync --full</code></pre>
<table><caption>RTO</caption><tr><th>Component</th><th>Minutes</th></tr><tr><td>Postgres</td><td>10</td></tr></table>
<img src="a.png" alt="diagram">
<div><p>Trailing paragraph inside div.</p></div></main>
<footer>© 2026 ACME</footer><script>alert(1)</script></body></html>"""


def fetched(
    html: str, uri: str = "https://wiki.example.com/runbooks/restore?utm=1"
) -> FetchedObject:
    item = SourceItem(
        uri=uri, canonical_uri="https://wiki.example.com/runbooks/restore", source_type="html"
    )
    return FetchedObject.from_bytes(item, html.encode(), mime_type="text/html")


def test_html_parser_blocks_title_canonical_and_chrome_removal() -> None:
    doc = HtmlParser().parse(fetched(PAGE))
    assert doc.title == "Runbook: Restore"
    assert doc.source_uri == "https://wiki.example.com/runbooks/restore"
    assert "Home" not in doc.text
    assert "ACME" not in doc.text
    assert "alert" not in doc.text
    kinds = [b.kind for b in doc.blocks]
    assert kinds == [
        "heading",
        "paragraph",
        "heading",
        "list",
        "code",
        "table",
        "image",
        "paragraph",
    ]
    assert (
        doc.blocks[1].text == "Restore MinIO first, then `pg_restore`. See [docs](https://x.y/z)."
    )  # type: ignore[union-attr]
    lst = next(b for b in doc.blocks if isinstance(b, ListBlock))
    assert lst.ordered is True
    assert lst.items == ("Stop API", "Restore volume")
    code = next(b for b in doc.blocks if isinstance(b, CodeBlock))
    assert code.language == "bash"
    table = next(b for b in doc.blocks if isinstance(b, TableBlock))
    assert table.caption == "RTO"
    assert table.n_rows == 1
    assert doc.metadata["html.description"] == "How to restore"


def test_html_parser_densest_subtree_fallback() -> None:
    body = (
        "<html><body><div id='side'>"
        + "<a href='/l'>link</a> " * 50
        + "</div><div id='c'><p>"
        + "content word " * 80
        + "</p></div></body></html>"
    )
    doc = HtmlParser().parse(fetched(body))
    assert "content word" in doc.text
    assert "link link" not in doc.text


PAGES = {
    "/robots.txt": (200, "User-agent: *\nDisallow: /private/\n", "text/plain"),
    "/sitemap.xml": (
        200,
        '<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://w.com/a</loc></url><url><loc>https://w.com/b</loc></url><url><loc>https://w.com/private/x</loc></url></urlset>',
        "application/xml",
    ),
    "/a": (
        200,
        "<html><body><main><h1>A</h1><p>"
        + "alpha " * 60
        + "</p><a href='/c'>c</a></main></body></html>",
        "text/html",
    ),
    "/b": (200, "<html><body><main><p>" + "beta " * 60 + "</p></main></body></html>", "text/html"),
    "/c": (200, "<html><body><main><p>" + "gamma " * 60 + "</p></main></body></html>", "text/html"),
    "/private/x": (200, "<html><body>secret</body></html>", "text/html"),
}


def _transport(
    pages: dict[str, tuple[int, str, str]], calls: list[tuple[str, str | None]]
) -> httpx.MockTransport:
    def handler(req: httpx.Request) -> httpx.Response:
        calls.append((req.url.path, req.headers.get("if-none-match")))
        p = req.url.path
        if p == "/b" and req.headers.get("if-none-match") == '"b1"':
            return httpx.Response(304)
        if p not in pages:
            return httpx.Response(404)
        status, body, ctype = pages[p]
        headers = {"content-type": ctype}
        if p == "/b":
            headers["etag"] = '"b1"'
        return httpx.Response(status, content=body.encode(), headers=headers)

    return httpx.MockTransport(handler)


def _cfg() -> HtmlConnectorConfig:
    return HtmlConnectorConfig(
        id="html-test",
        type="html",
        sitemap_url="https://w.com/sitemap.xml",
        seed_urls=["https://w.com/a"],
        requests_per_second=10_000,
    )


def test_html_connector_sitemap_crawl_robots_conditional_get_and_deletion() -> None:
    pages = dict(PAGES)
    calls: list[tuple[str, str | None]] = []
    conn = HtmlConnector(_cfg(), transport=_transport(pages, calls))
    events = list(conn.discover({}))
    items = [e for e in events if isinstance(e, SourceItem)]
    assert [i.uri for i in items] == [
        "https://w.com/a",
        "https://w.com/b",
        "https://w.com/c",
    ]  # /private blocked by robots
    fetched_objs = [conn.fetch(i) for i in items]
    assert fetched_objs[1].source_metadata["http.etag"] == '"b1"'
    state = conn.checkpoint({}, fetched_objs)
    assert sorted(state["urls"]) == ["https://w.com/a", "https://w.com/b", "https://w.com/c"]

    del pages["/c"]
    calls.clear()
    conn2 = HtmlConnector(_cfg(), transport=_transport(pages, calls))
    events2 = list(conn2.discover(state))
    assert [e.uri for e in events2 if isinstance(e, SourceItem)] == [
        "https://w.com/a"
    ]  # b → 304, c → gone
    deletions = [e for e in events2 if isinstance(e, DeletionEvent)]
    assert [d.canonical_uri for d in deletions] == ["https://w.com/c"]
    assert any(path == "/b" and etag == '"b1"' for path, etag in calls)
    state2 = conn2.checkpoint(state, [conn2.fetch(e) for e in events2 if isinstance(e, SourceItem)])
    assert "https://w.com/c" not in state2["urls"]
