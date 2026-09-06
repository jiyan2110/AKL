"""Tracing (PRD §8.4): OpenTelemetry spans for API requests and retrieval stages.

Disabled by default (``AKL_OTEL_ENABLED=false``) so tests and offline runs never need a
collector. When enabled, spans are exported over OTLP/HTTP to ``otel_exporter_endpoint``
(e.g. an OpenTelemetry Collector feeding Tempo/Jaeger). ``traced()`` is a no-op context manager
when tracing is off, so call sites never need an ``if enabled`` check.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

from akl.config import Settings

_configured = False


def configure_tracing(settings: Settings) -> None:
    """Idempotent. No-op unless ``observability.otel_enabled`` is true."""
    global _configured
    if _configured or not settings.observability.otel_enabled:
        return
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    resource = Resource.create(
        {SERVICE_NAME: settings.core.service_name, "service.env": settings.core.env.value}
    )
    provider = TracerProvider(
        resource=resource, sampler=TraceIdRatioBased(settings.observability.otel_sample_ratio)
    )
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.observability.otel_exporter_endpoint))
    )
    trace.set_tracer_provider(provider)
    _configured = True


def get_tracer(name: str = "akl") -> trace.Tracer:
    return trace.get_tracer(name)


@contextmanager
def traced(name: str, **attributes: Any) -> Iterator[trace.Span]:
    """Start a span named ``name`` with ``attributes``; a true no-op span when tracing is disabled."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(
                    key, value if isinstance(value, str | bool | int | float) else str(value)
                )
        yield span


def record_exception(span: trace.Span, exc: BaseException) -> None:
    span.record_exception(exc)
    span.set_status(trace.StatusCode.ERROR, str(exc))
