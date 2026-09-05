"""HTML connector (PRD §3.3.3): sitemap, URL list or bounded crawl over httpx.

* Discovery issues conditional ``GET`` requests (``If-None-Match`` / ``If-Modified-Since``)
  using the ETag/Last-Modified stored per URL in the connector state; ``304`` → unchanged.
* ``robots.txt`` is honoured; per-host rate limit; same-host + allow/deny patterns for crawl.
* URLs that were known but now return 404/410 become :class:`DeletionEvent`s.
"""

from __future__ import annotations

import re
import threading
import time
import urllib.robotparser
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, ClassVar, Literal
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree as ET

import httpx
from pydantic import Field
from selectolax.parser import HTMLParser as _Selectolax

from akl import ids
from akl.ingestion.connectors.base import BaseConnector, ConnectorConfig, ConnectorError
from akl.ingestion.models import (
    ConnectorHealth,
    DeletionEvent,
    FetchedObject,
    SourceItem,
    SourceType,
)

DEFAULT_UA = "AKL-Crawler/1.0 (+https://akl.internal/contact)"


class HtmlConnectorConfig(ConnectorConfig):
    type: Literal["html"] = "html"
    sitemap_url: str | None = None
    urls: list[str] = Field(default_factory=list)
    seed_urls: list[str] = Field(default_factory=list)
    allow_patterns: list[str] = Field(default_factory=list)  # regex on full URL (crawl)
    deny_patterns: list[str] = Field(
        default_factory=lambda: [r"\.(png|jpe?g|gif|svg|css|js|zip|pdf)(\?|$)"]
    )
    max_depth: int = Field(default=2, ge=0, le=6)
    max_pages: int = Field(default=500, ge=1)
    same_host_only: bool = True
    requests_per_second: float = Field(default=2.0, gt=0)
    timeout_s: float = Field(default=20.0, gt=0)
    user_agent: str = DEFAULT_UA
    respect_robots: bool = True


class _RateLimiter:
    def __init__(self, rps: float) -> None:
        self._interval = 1.0 / rps
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str) -> None:
        with self._lock:
            now = time.monotonic()
            nxt = self._last.get(host, 0.0) + self._interval
            delay = max(0.0, nxt - now)
            self._last[host] = max(now, nxt)
        if delay:
            time.sleep(delay)


class HtmlConnector(BaseConnector):
    name = "html"
    version = "1.0.0"
    source_type: SourceType = "html"
    config_cls: ClassVar[type[ConnectorConfig]] = HtmlConnectorConfig

    def __init__(
        self, config: ConnectorConfig, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        if not isinstance(config, HtmlConnectorConfig):
            config = HtmlConnectorConfig.model_validate(config.model_dump())
        super().__init__(config)
        self.cfg: HtmlConnectorConfig = config
        self._client = httpx.Client(
            headers={
                "User-Agent": self.cfg.user_agent,
                "Accept": "text/html,application/xhtml+xml",
            },
            timeout=self.cfg.timeout_s,
            follow_redirects=True,
            transport=transport,
        )
        self._limiter = _RateLimiter(self.cfg.requests_per_second)
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._allow = [re.compile(p) for p in self.cfg.allow_patterns]
        self._deny = [re.compile(p) for p in self.cfg.deny_patterns]
        self._gone: set[str] = set()
        self._responses: dict[str, tuple[bytes, dict[str, str]]] = {}

    # -- HTTP -------------------------------------------------------------------------
    def _get(self, url: str, headers: Mapping[str, str] | None = None) -> httpx.Response:
        self._limiter.wait(urlparse(url).netloc)
        try:
            return self._client.get(url, headers=dict(headers or {}))
        except httpx.HTTPError as exc:
            raise ConnectorError(f"GET {url} failed: {exc}", details={"url": url}) from exc

    def _allowed_by_robots(self, url: str) -> bool:
        if not self.cfg.respect_robots:
            return True
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                resp = self._get(f"{base}/robots.txt")
                rp.parse(resp.text.splitlines() if resp.status_code == 200 else [])
            except ConnectorError:
                rp.parse([])
            self._robots[base] = rp
        rp2 = self._robots[base]
        return rp2 is None or rp2.can_fetch(self.cfg.user_agent, url)

    def _url_ok(self, url: str, seed_host: str | None) -> bool:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False
        if self.cfg.same_host_only and seed_host and p.netloc.lower() != seed_host.lower():
            return False
        if any(r.search(url) for r in self._deny):
            return False
        return not self._allow or any(r.search(url) for r in self._allow)

    # -- discovery sources ---------------------------------------------------------------
    def _from_sitemap(self, url: str, depth: int = 0) -> list[str]:
        if depth > 2:
            return []
        resp = self._get(url)
        if resp.status_code != 200:
            raise ConnectorError(f"sitemap {url} returned {resp.status_code}", details={"url": url})
        try:
            root = ET.fromstring(resp.content)  # noqa: S314 - sitemap XML from configured host
        except ET.ParseError as exc:
            raise ConnectorError(f"sitemap {url} is not valid XML", details={"url": url}) from exc
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls: list[str] = []
        for loc in root.findall("sm:sitemap/sm:loc", ns):
            if loc.text:
                urls.extend(self._from_sitemap(loc.text.strip(), depth + 1))
        urls.extend(loc.text.strip() for loc in root.findall("sm:url/sm:loc", ns) if loc.text)
        return urls

    def _crawl(self) -> list[str]:
        found: list[str] = []
        seen: set[str] = set()
        queue: deque[tuple[str, int]] = deque((u, 0) for u in self.cfg.seed_urls)
        while queue and len(found) < self.cfg.max_pages:
            url, depth = queue.popleft()
            canon = ids.canonicalize_uri(url)
            if canon in seen:
                continue
            seen.add(canon)
            seed_host = urlparse(url).netloc if self.cfg.same_host_only else None
            if not self._url_ok(url, seed_host) or not self._allowed_by_robots(url):
                continue
            resp = self._get(url)
            if resp.status_code != 200 or "html" not in resp.headers.get("content-type", ""):
                continue
            found.append(url)
            self._responses[canon] = (resp.content, dict(resp.headers))
            if depth < self.cfg.max_depth:
                tree = _Selectolax(resp.text)
                for a in tree.css("a[href]"):
                    href = urljoin(url, (a.attributes.get("href") or "").split("#", 1)[0])
                    if self._url_ok(href, seed_host):
                        queue.append((href, depth + 1))
        return found

    # -- contract -------------------------------------------------------------------------
    def discover(self, state: Mapping[str, Any]) -> Iterator[SourceItem | DeletionEvent]:
        known: dict[str, dict[str, str]] = dict(state.get("urls", {}))
        candidates: list[str] = []
        if self.cfg.sitemap_url:
            candidates.extend(self._from_sitemap(self.cfg.sitemap_url))
        candidates.extend(self.cfg.urls)
        if self.cfg.seed_urls:
            candidates.extend(self._crawl())

        seen: set[str] = set()
        for url in candidates:
            canon = ids.canonicalize_uri(url)
            if canon in seen:
                continue
            seen.add(canon)
            if not self._allowed_by_robots(url):
                continue
            prior = known.get(canon, {})
            if canon not in self._responses:
                headers: dict[str, str] = {}
                if prior.get("etag"):
                    headers["If-None-Match"] = prior["etag"]
                if prior.get("last_modified"):
                    headers["If-Modified-Since"] = prior["last_modified"]
                resp = self._get(url, headers)
                if resp.status_code == 304:
                    continue
                if resp.status_code in (404, 410):
                    if canon in known:
                        self._gone.add(canon)
                        yield DeletionEvent(canonical_uri=canon, reason=f"http_{resp.status_code}")
                    continue
                if resp.status_code != 200:
                    continue
                self._responses[canon] = (resp.content, dict(resp.headers))
            level, groups = self.cfg.resolve_security(urlparse(url).path)
            yield SourceItem(
                uri=url,
                canonical_uri=canon,
                source_type="html",
                filename=(urlparse(url).path.rstrip("/").rsplit("/", 1)[-1] or "index") + ".html",
                security_level=level,
                allowed_groups=groups,
                source_metadata={"host": urlparse(url).netloc},
            )
        for canon in set(known) - seen:
            self._gone.add(canon)
            yield DeletionEvent(canonical_uri=canon, reason="not_in_source_listing")

    def fetch(self, item: SourceItem) -> FetchedObject:
        cached = self._responses.pop(item.canonical_uri, None)
        if cached is None:
            resp = self._get(item.uri)
            if resp.status_code != 200:
                raise ConnectorError(
                    f"GET {item.uri} -> {resp.status_code}",
                    details={"url": item.uri},
                    retryable=resp.status_code >= 500,
                )
            cached = (resp.content, dict(resp.headers))
        data, headers = cached
        meta = {k: headers[k] for k in ("etag", "last-modified", "content-type") if k in headers}
        return FetchedObject.from_bytes(
            item,
            data,
            mime_type=(headers.get("content-type") or "text/html").split(";")[0].strip(),
            source_metadata={
                "http.etag": meta.get("etag", ""),
                "http.last_modified": meta.get("last-modified", ""),
            },
        )

    def checkpoint(
        self, state: Mapping[str, Any], committed: Sequence[FetchedObject]
    ) -> dict[str, Any]:
        urls: dict[str, dict[str, str]] = dict(state.get("urls", {}))
        for obj in committed:
            urls[obj.item.canonical_uri] = {
                "etag": obj.source_metadata.get("http.etag", ""),
                "last_modified": obj.source_metadata.get("http.last_modified", ""),
                "sha256": obj.sha256,
            }
        for canon in self._gone:
            urls.pop(canon, None)
        return {"urls": urls, "last_run": time.time()}

    def health(self) -> ConnectorHealth:
        candidates = [self.cfg.sitemap_url, *self.cfg.urls, *self.cfg.seed_urls]
        target: str | None = next((c for c in candidates if c), None)
        if not target:
            return ConnectorHealth(
                ok=False, latency_ms=0.0, detail="no sitemap_url/urls/seed_urls configured"
            )
        start = time.perf_counter()
        try:
            resp = self._client.head(target)
            return ConnectorHealth(
                ok=resp.status_code < 400,
                latency_ms=(time.perf_counter() - start) * 1000,
                detail=f"HEAD {target} -> {resp.status_code}",
            )
        except httpx.HTTPError as exc:
            return ConnectorHealth(
                ok=False, latency_ms=(time.perf_counter() - start) * 1000, detail=str(exc)
            )

    def close(self) -> None:
        self._client.close()
