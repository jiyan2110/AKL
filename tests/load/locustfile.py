"""Load test (PRD §11.5): weighted traffic against the search/chat/upload endpoints.

Run against a real, running API (not spun up by this file):
    AKL_LOAD_TARGET=http://localhost:8000 AKL_LOAD_API_KEY=akl_... \
      uv run --group load locust -f tests/load/locustfile.py --headless -u 20 -r 2 -t 2m
Or `make load-test` for sane defaults. Auth: set AKL_LOAD_API_KEY (preferred) or
AKL_LOAD_JWT; with neither, requests go out unauthenticated (fine only if the target has
AKL_AUTH_DISABLED=true, e.g. a local dev stack).
"""

from __future__ import annotations

import os
import random

from locust import HttpUser, between, task

QUERIES = [
    "how are nightly postgres backups taken",
    "what does make seed do",
    "how is the qdrant collection synced",
    "what happens during a hard delete",
    "how does the embedding cache work",
    "what is the chunk_config_hash",
]


def _auth_headers() -> dict[str, str]:
    api_key = os.environ.get("AKL_LOAD_API_KEY")
    jwt = os.environ.get("AKL_LOAD_JWT")
    if api_key:
        return {"X-API-Key": api_key}
    if jwt:
        return {"Authorization": f"Bearer {jwt}"}
    return {}


class AklUser(HttpUser):
    wait_time = between(0.5, 2.5)

    def on_start(self) -> None:
        self.headers = _auth_headers()

    @task(6)
    def search(self) -> None:
        query = random.choice(QUERIES)  # noqa: S311 - load generation, not security
        self.client.post(
            "/v1/search", json={"query": query, "k": 8}, headers=self.headers, name="/v1/search"
        )

    @task(3)
    def chat(self) -> None:
        query = random.choice(QUERIES)  # noqa: S311
        self.client.post("/v1/chat", json={"query": query}, headers=self.headers, name="/v1/chat")

    @task(1)
    def chat_stream(self) -> None:
        query = random.choice(QUERIES)  # noqa: S311
        with self.client.post(
            "/v1/chat",
            json={"query": query, "stream": True},
            headers=self.headers,
            name="/v1/chat (stream)",
            stream=True,
            catch_response=True,
        ) as resp:
            for _ in resp.iter_lines():
                pass  # drain the SSE stream so the connection timing reflects a real client
            resp.success() if resp.status_code == 200 else resp.failure(
                f"status {resp.status_code}"
            )

    @task(1)
    def health(self) -> None:
        self.client.get("/v1/health", name="/v1/health")
