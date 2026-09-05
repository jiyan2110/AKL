"""In-memory token-bucket rate limiter per principal and route class (PRD §10.1).

MVP implementation for a single API process; the Postgres/Redis-backed limiter (Appendix A.14
``rate_limit_buckets``) replaces the store when multiple replicas are deployed.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from akl.errors import AKLError


class RateLimitedError(AKLError):
    code = "AKL-E1006"
    http_status = 429
    retryable = True


@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBucketLimiter:
    def __init__(self, limits_per_minute: dict[str, int], *, default_rpm: int) -> None:
        self._limits = dict(limits_per_minute)
        self._default = default_rpm
        self._buckets: dict[tuple[str, str], _Bucket] = {}
        self._lock = threading.Lock()

    def limit_for(self, route_class: str) -> int:
        return self._limits.get(route_class, self._default)

    def check(self, principal_id: str, route_class: str) -> tuple[bool, float]:
        """Return ``(allowed, retry_after_seconds)``."""
        rpm = self.limit_for(route_class)
        rate = rpm / 60.0
        now = time.monotonic()
        with self._lock:
            key = (principal_id, route_class)
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=float(rpm), updated=now)
                self._buckets[key] = bucket
            bucket.tokens = min(float(rpm), bucket.tokens + (now - bucket.updated) * rate)
            bucket.updated = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0
            return False, round((1.0 - bucket.tokens) / rate, 2)

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()
