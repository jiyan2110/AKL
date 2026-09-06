"""Structured logging (PRD §8.1): structlog → JSON in prod, console in dev; redaction; context ids.

``configure_logging()`` is called once at process start (API, CLI, and task_scope for Airflow
tasks). ``bind_context`` attaches ``request_id``/``run_id``/``task_id`` to every log line emitted
for the duration of a request or task via structlog's contextvars, so log lines can be joined
across a request without threading the id through every call site.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any

import structlog

from akl.config import Environment, Settings

_SECRET_KEYS = re.compile(
    r"(password|secret|token|api[_-]?key|authorization|jwt|dsn)", re.IGNORECASE
)
_BEARER = re.compile(r"Bearer\s+[A-Za-z0-9\-_.=]+", re.IGNORECASE)
_URL_CREDS = re.compile(r"(://)([^/@\s]+):([^/@\s]+)(@)")
REDACTED = "***REDACTED***"


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        value = _BEARER.sub(f"Bearer {REDACTED}", value)
        value = _URL_CREDS.sub(rf"\1{REDACTED}:{REDACTED}\4", value)
        return value
    if isinstance(value, dict):
        return {
            k: (REDACTED if _SECRET_KEYS.search(str(k)) else _redact_value(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    return value


def redact_processor(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """structlog processor: never let secrets or bearer tokens reach a log sink."""
    for key in list(event_dict):
        if _SECRET_KEYS.search(key):
            event_dict[key] = REDACTED
        else:
            event_dict[key] = _redact_value(event_dict[key])
    return event_dict


def configure_logging(settings: Settings) -> None:
    """Idempotent: safe to call from the API lifespan, the CLI entrypoint, and each Airflow task."""
    level = getattr(logging, settings.core.log_level.value, logging.INFO)
    logging.basicConfig(level=level, format="%(message)s", force=True)
    for name in ("uvicorn.access",):
        logging.getLogger(name).setLevel(
            logging.WARNING
        )  # superseded by RequestContextMiddleware's line

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        redact_processor,
        structlog.processors.add_log_level,
    ]
    renderer = (
        structlog.processors.JSONRenderer()
        if settings.core.env is not Environment.DEV
        else structlog.dev.ConsoleRenderer(colors=True)
    )
    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)


@contextmanager
def bind_context(**kwargs: Any) -> Iterator[None]:
    """Bind ids (request_id, run_id, task_id, principal, dag_id, ...) for every log line in scope."""
    tokens = structlog.contextvars.bind_contextvars(
        **{k: v for k, v in kwargs.items() if v is not None}
    )
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)
