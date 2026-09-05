"""GitHub repository connector (PRD §3.3.4).

API mode (default): compare branch head with the checkpointed commit; on change
fetch the recursive tree once, diff ``{path: blob_sha}`` against the previous
snapshot and download only added/modified blobs. Deleted paths produce
:class:`DeletionEvent`s. Clone mode (``git clone --depth 1``) is the fallback
for truncated trees or exhausted rate limits.
"""

from __future__ import annotations

import base64
import fnmatch
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

import httpx
from pydantic import Field

from akl import ids
from akl.ingestion.connectors.base import BaseConnector, ConnectorConfig, ConnectorError
from akl.ingestion.models import (
    ConnectorHealth,
    DeletionEvent,
    FetchedObject,
    SourceItem,
    SourceType,
)

DEFAULT_INCLUDE = ["**/*.md", "**/*.mdx", "**/*.rst", "**/*.txt", "docs/**"]
DEFAULT_EXCLUDE = ["**/node_modules/**", "**/vendor/**", "**/CHANGELOG*", "**/.github/**"]
DEFAULT_CODE = [
    "**/*.py",
    "**/*.ts",
    "**/*.js",
    "**/*.go",
    "**/*.java",
    "**/*.rs",
    "**/*.sql",
    "**/*.yaml",
    "**/*.yml",
]
_EXT_MIME = {
    "md": "text/markdown",
    "mdx": "text/markdown",
    "markdown": "text/markdown",
    "rst": "text/x-rst",
    "txt": "text/plain",
    "html": "text/html",
    "htm": "text/html",
}


class GitHubRateLimitError(ConnectorError):
    code = "AKL-E3040"


class GitHubNotFoundError(ConnectorError):
    code = "AKL-E3041"
    retryable = False


class GitHubConnectorConfig(ConnectorConfig):
    type: Literal["github"] = "github"
    owner: str
    repo: str
    branch: str = "main"
    include_globs: list[str] = Field(default_factory=lambda: list(DEFAULT_INCLUDE))
    exclude_globs: list[str] = Field(default_factory=lambda: list(DEFAULT_EXCLUDE))
    include_code: bool = False
    code_globs: list[str] = Field(default_factory=lambda: list(DEFAULT_CODE))
    max_file_bytes: int = Field(default=1_048_576, ge=1)
    api_url: str = "https://api.github.com"
    token_env: str = "AKL_GITHUB_TOKEN"  # noqa: S105 - env var NAME, not a secret
    mode: Literal["api", "clone", "auto"] = "auto"
    rate_limit_floor: int = Field(default=50, ge=0)

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"


def _glob(rel: str, pattern: str) -> bool:
    return fnmatch.fnmatch(rel, pattern) or (
        pattern.startswith("**/") and fnmatch.fnmatch(rel, pattern[3:])
    )


class GitHubConnector(BaseConnector):
    name = "github"
    version = "1.0.0"
    source_type: SourceType = "github"
    config_cls: ClassVar[type[ConnectorConfig]] = GitHubConnectorConfig

    def __init__(
        self, config: ConnectorConfig, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        if not isinstance(config, GitHubConnectorConfig):
            config = GitHubConnectorConfig.model_validate(config.model_dump())
        super().__init__(config)
        self.cfg: GitHubConnectorConfig = config
        token = os.environ.get(self.cfg.token_env) or self._token_file()
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "akl-github-connector/1.0",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(
            base_url=self.cfg.api_url, headers=headers, timeout=30.0, transport=transport
        )
        self.rate_limit_remaining: int | None = None
        self._head_sha: str | None = None
        self._tree: dict[str, tuple[str, int]] = {}  # path -> (blob_sha, size)
        self._deleted: list[str] = []
        self._clone_dir: Path | None = None
        self.snapshot_rows: list[dict[str, Any]] = []

    def _token_file(self) -> str | None:
        path = os.environ.get(f"{self.cfg.token_env}_FILE")
        return (
            Path(path).read_text(encoding="utf-8").strip() if path and Path(path).exists() else None
        )

    # -- API helpers -------------------------------------------------------------------
    def _get(self, path: str, **params: Any) -> httpx.Response:
        try:
            resp = self._client.get(path, params=params or None)
        except httpx.HTTPError as exc:
            raise ConnectorError(
                f"GitHub GET {path} failed: {exc}", details={"path": path}
            ) from exc
        remaining = resp.headers.get("x-ratelimit-remaining")
        if remaining is not None and remaining.isdigit():
            self.rate_limit_remaining = int(remaining)
        if resp.status_code == 404:
            raise GitHubNotFoundError(f"GitHub 404: {path}", details={"path": path})
        if resp.status_code in (403, 429) and self.rate_limit_remaining == 0:
            reset = int(resp.headers.get("x-ratelimit-reset", "0"))
            raise GitHubRateLimitError(
                "GitHub rate limit exhausted", details={"reset_epoch": reset}
            )
        if resp.status_code >= 400:
            raise ConnectorError(
                f"GitHub {resp.status_code}: {path}",
                details={"path": path, "body": resp.text[:200]},
                retryable=resp.status_code >= 500,
            )
        return resp

    def head_sha(self) -> str:
        if self._head_sha is None:
            data = self._get(f"/repos/{self.cfg.full_name}/branches/{self.cfg.branch}").json()
            self._head_sha = str(data["commit"]["sha"])
        return self._head_sha

    def fetch_tree(self, sha: str) -> tuple[dict[str, tuple[str, int]], bool]:
        data = self._get(f"/repos/{self.cfg.full_name}/git/trees/{sha}", recursive="1").json()
        tree = {
            e["path"]: (str(e["sha"]), int(e.get("size", 0)))
            for e in data.get("tree", [])
            if e.get("type") == "blob"
        }
        return tree, bool(data.get("truncated"))

    def _wants(self, path: str) -> bool:
        if any(_glob(path, p) for p in self.cfg.exclude_globs):
            return False
        if any(_glob(path, p) for p in self.cfg.include_globs):
            return True
        return self.cfg.include_code and any(_glob(path, p) for p in self.cfg.code_globs)

    def _uri(self, path: str, sha: str | None = None) -> str:
        return f"github://{self.cfg.owner}/{self.cfg.repo}/{sha or self.cfg.branch}/{path}"

    # -- clone mode --------------------------------------------------------------------
    def _clone(self) -> Path:
        if self._clone_dir is None:
            if shutil.which("git") is None:
                raise ConnectorError("git binary not available for clone mode", retryable=False)
            target = Path(tempfile.mkdtemp(prefix="akl-gh-"))
            url = f"https://github.com/{self.cfg.full_name}.git"
            token = os.environ.get(self.cfg.token_env)
            if token:
                url = f"https://x-access-token:{token}@github.com/{self.cfg.full_name}.git"
            cmd = [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                self.cfg.branch,
                "--quiet",
                url,
                str(target),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=600)  # noqa: S603 - fixed argv, no shell
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                raise ConnectorError(
                    "git clone failed", details={"repo": self.cfg.full_name, "error": str(exc)}
                ) from exc
            self._clone_dir = target
        return self._clone_dir

    @staticmethod
    def _git(repo: Path, *args: str) -> str:
        git = shutil.which("git") or "git"
        cmd = [git, "-C", str(repo), *args]
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout  # noqa: S603

    def _tree_from_clone(self) -> tuple[str, dict[str, tuple[str, int]]]:
        repo = self._clone()
        sha = self._git(repo, "rev-parse", "HEAD").strip()
        out = self._git(repo, "ls-tree", "-r", "-l", "HEAD")
        tree: dict[str, tuple[str, int]] = {}
        for line in out.splitlines():
            meta, _, path = line.partition("\t")
            parts = meta.split()
            if len(parts) >= 4 and parts[1] == "blob":
                tree[path] = (parts[2], int(parts[3]) if parts[3] != "-" else 0)
        return sha, tree

    # -- contract ---------------------------------------------------------------------------
    def discover(self, state: Mapping[str, Any]) -> Iterator[SourceItem | DeletionEvent]:
        previous_sha = state.get("last_commit_sha")
        previous_tree: dict[str, str] = dict(state.get("tree", {}))
        use_clone = self.cfg.mode == "clone"
        if not use_clone:
            try:
                sha = self.head_sha()
                if sha == previous_sha:
                    self._tree = {p: (s, 0) for p, s in previous_tree.items()}
                    return
                tree, truncated = self.fetch_tree(sha)
                if truncated and self.cfg.mode == "auto":
                    use_clone = True
            except GitHubRateLimitError:
                if self.cfg.mode != "auto":
                    raise
                use_clone = True
        if use_clone:
            sha, tree = self._tree_from_clone()
            if sha == previous_sha:
                self._tree = {p: (s, 0) for p, s in previous_tree.items()}
                return
        self._head_sha = sha
        self._tree = {
            p: v for p, v in tree.items() if self._wants(p) and v[1] <= self.cfg.max_file_bytes
        }
        now = time.time()
        self.snapshot_rows = [
            {
                "repo": self.cfg.full_name,
                "commit_sha": sha,
                "path": p,
                "blob_sha": s,
                "size_bytes": size,
                "mode": "blob",
                "snapshot_at": now,
            }
            for p, (s, size) in sorted(self._tree.items())
        ]
        for path, (blob_sha, size) in sorted(self._tree.items()):
            if previous_tree.get(path) == blob_sha:
                continue
            level, groups = self.cfg.resolve_security(path)
            uri = self._uri(path, sha)
            yield SourceItem(
                uri=uri,
                canonical_uri=ids.canonicalize_uri(self._uri(path)),
                source_type="github",
                filename=path.rsplit("/", 1)[-1],
                expected_size=size or None,
                hint_hash=blob_sha,
                security_level=level,
                allowed_groups=groups,
                source_metadata={
                    "git.repo": self.cfg.full_name,
                    "git.branch": self.cfg.branch,
                    "git.path": path,
                    "git.commit_sha": sha,
                    "git.blob_sha": blob_sha,
                },
            )
        self._deleted = sorted(p for p in previous_tree if p not in self._tree)
        for path in self._deleted:
            yield DeletionEvent(
                canonical_uri=ids.canonicalize_uri(self._uri(path)), reason="removed_from_repo"
            )

    def fetch(self, item: SourceItem) -> FetchedObject:
        path = item.source_metadata["git.path"]
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        mime = _EXT_MIME.get(ext, "text/plain")
        if self._clone_dir is not None:
            data = (self._clone_dir / path).read_bytes()
        else:
            if (
                self.rate_limit_remaining is not None
                and self.rate_limit_remaining < self.cfg.rate_limit_floor
            ):
                raise GitHubRateLimitError(
                    "GitHub rate limit near exhaustion; deferring fetch",
                    details={"remaining": self.rate_limit_remaining},
                )
            blob = self._get(f"/repos/{self.cfg.full_name}/git/blobs/{item.hint_hash}").json()
            content = blob.get("content", "")
            data = (
                base64.b64decode(content)
                if blob.get("encoding") == "base64"
                else str(content).encode()
            )
        return FetchedObject.from_bytes(item, data, mime_type=mime)

    def checkpoint(
        self, state: Mapping[str, Any], committed: Sequence[FetchedObject]
    ) -> dict[str, Any]:
        tree: dict[str, str] = dict(state.get("tree", {}))
        for obj in committed:
            tree[obj.item.source_metadata["git.path"]] = obj.item.source_metadata["git.blob_sha"]
        for path in self._deleted:
            tree.pop(path, None)
        new_state: dict[str, Any] = {"tree": tree, "last_run": time.time()}
        # Advance the commit pointer only when every wanted blob at HEAD is checkpointed.
        if self._head_sha and all(tree.get(p) == s for p, (s, _) in self._tree.items()):
            new_state["last_commit_sha"] = self._head_sha
        elif state.get("last_commit_sha"):
            new_state["last_commit_sha"] = state["last_commit_sha"]
        if self._clone_dir is not None:
            shutil.rmtree(self._clone_dir, ignore_errors=True)
            self._clone_dir = None
        return new_state

    def health(self) -> ConnectorHealth:
        start = time.perf_counter()
        try:
            self.head_sha()
            return ConnectorHealth(
                ok=True,
                latency_ms=(time.perf_counter() - start) * 1000,
                detail=f"{self.cfg.full_name}@{self.cfg.branch} rate_remaining={self.rate_limit_remaining}",
            )
        except ConnectorError as exc:
            return ConnectorHealth(
                ok=False, latency_ms=(time.perf_counter() - start) * 1000, detail=str(exc)
            )
