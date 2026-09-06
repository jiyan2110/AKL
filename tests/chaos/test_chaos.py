"""Chaos tests (PRD §11.6): stop a real dependency container, confirm the API degrades gracefully
(a labelled flag / 503, never an unhandled 500 or a hang) rather than crashing, then restart it and
confirm recovery. These actually stop containers in your Compose stack via the Docker CLI — opt-in
only, and skipped unless both conditions hold:

* ``AKL_ALLOW_CHAOS_TESTS=true`` is set (this is destructive to whatever stack you point it at)
* the ``docker`` CLI is on PATH and ``docker compose ps`` succeeds against the compose files in
  this repo (so it silently skips in CI unless a runner explicitly opts in — see ``make chaos``)

Each test restores the stopped service in a ``finally`` block and waits for its healthcheck before
returning control, so a failed assertion never leaves the stack degraded for the next test.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Iterator

import httpx
import pytest

pytestmark = pytest.mark.chaos

COMPOSE = ["docker", "compose", "-f", "docker-compose.yml", "-f", "docker-compose.dev.yml"]
API_BASE = os.environ.get("AKL_CHAOS_API_URL", "http://localhost:8000")
HEALTH_TIMEOUT_S = 60


def _chaos_enabled() -> bool:
    if os.environ.get("AKL_ALLOW_CHAOS_TESTS", "").lower() != "true":
        return False
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run([*COMPOSE, "ps"], check=True, capture_output=True, timeout=15)  # noqa: S603 - fixed argv, no user input
    except (subprocess.SubprocessError, OSError):
        return False
    return True


requires_chaos = pytest.mark.skipif(
    not _chaos_enabled(),
    reason="set AKL_ALLOW_CHAOS_TESTS=true with a running Docker Compose stack to run chaos tests",
)


def _stop(service: str) -> None:
    subprocess.run([*COMPOSE, "stop", service], check=True, capture_output=True, timeout=60)  # noqa: S603


def _start(service: str) -> None:
    subprocess.run([*COMPOSE, "start", service], check=True, capture_output=True, timeout=60)  # noqa: S603


def _wait_healthy(service: str, *, timeout_s: int = HEALTH_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = subprocess.run(  # noqa: S603
            [*COMPOSE, "ps", "--format", "json", service],
            check=False,
            capture_output=True,
            timeout=15,
            text=True,
        )
        if '"Health":"healthy"' in result.stdout or '"State":"running"' in result.stdout:
            return
        time.sleep(2)
    raise TimeoutError(f"{service} did not become healthy within {timeout_s}s")


@pytest.fixture
def api() -> Iterator[httpx.Client]:
    with httpx.Client(base_url=API_BASE, timeout=15.0) as client:
        try:
            client.get("/v1/health").raise_for_status()
        except httpx.HTTPError as exc:
            pytest.skip(f"API not reachable at {API_BASE}: {exc}")
        yield client


@requires_chaos
def test_search_degrades_gracefully_without_qdrant(api: httpx.Client) -> None:
    """With Qdrant down, search must fall back to sparse-only (flagged), never crash or hang."""
    _stop("qdrant")
    try:
        deadline = time.monotonic() + 30
        resp = None
        while time.monotonic() < deadline:
            resp = api.post("/v1/search", json={"query": "how are documents stored", "k": 5})
            if resp.status_code != 200 or "dense_unavailable" in resp.json().get("flags", []):
                break
            time.sleep(1)  # a connection-pooled client may need a moment to notice qdrant is gone
        assert resp is not None
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            body = resp.json()
            assert "dense_unavailable" in body.get("flags", [])
            assert body.get("results") is not None  # sparse results can still come back
        health = api.get("/v1/health")
        assert health.status_code == 200  # liveness never depends on a single backend
    finally:
        _start("qdrant")
        _wait_healthy("qdrant")


@requires_chaos
def test_search_returns_503_without_postgres(api: httpx.Client) -> None:
    """With Postgres down, requests needing it must fail cleanly (503-class), not hang or 500."""
    _stop("postgres")
    try:
        deadline = time.monotonic() + 20
        resp = None
        while time.monotonic() < deadline:
            resp = api.post("/v1/search", json={"query": "how are documents stored", "k": 5})
            if resp.status_code != 200:
                break
            time.sleep(1)
        assert resp is not None
        assert resp.status_code < 500 or resp.status_code == 503
        assert resp.status_code != 500, (
            "an unhandled server error is a bug, not graceful degradation"
        )
    finally:
        _start("postgres")
        _wait_healthy("postgres")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if api.get("/v1/health/ready").status_code == 200:
                break
            time.sleep(2)


@requires_chaos
def test_api_recovers_after_dependency_restart(api: httpx.Client) -> None:
    """After Qdrant is stopped and restarted, search must return to fully-healthy (no stale flags)."""
    _stop("qdrant")
    try:
        time.sleep(3)
    finally:
        _start("qdrant")
        _wait_healthy("qdrant")
    deadline = time.monotonic() + 60
    recovered = False
    while time.monotonic() < deadline:
        resp = api.post("/v1/search", json={"query": "how are documents stored", "k": 5})
        if resp.status_code == 200 and "dense_unavailable" not in resp.json().get("flags", []):
            recovered = True
            break
        time.sleep(2)
    assert recovered, "search did not recover to a healthy state within 60s of qdrant restarting"
