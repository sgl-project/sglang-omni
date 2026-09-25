# SPDX-License-Identifier: Apache-2.0
"""Tracing state owned by coordinator and stage request lifetimes."""

import os
from collections.abc import Mapping
from typing import Literal

from opentelemetry import trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk import trace as sdk_trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

propagator = CompositePropagator(
    [TraceContextTextMapPropagator(), W3CBaggagePropagator()]
)


def create_tracer_provider(endpoint: str | None) -> sdk_trace.TracerProvider | None:
    if not endpoint:
        return None
    else:
        pass
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    protocol = os.environ.get(
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
        os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc"),
    )
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
    elif protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
    else:
        raise ValueError(f"Unsupported OTLP traces protocol: {protocol}")
    provider = sdk_trace.TracerProvider(
        resource=Resource.create(
            {"service.name": os.environ.get("OTEL_SERVICE_NAME", "sglang-omni")}
        )
    )
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    return provider


class RequestTraces:
    def __init__(
        self, tracer: trace.Tracer | None, name: str, stage: str | None = None
    ) -> None:
        self.tracer = tracer
        self.name = name
        self.stage = stage
        self.spans: dict[str, trace.Span] = {}
        self.carriers: dict[str, dict[str, str]] = {}
        self.first_outputs: set[str] = set()

    def start(self, request_id: str, headers: Mapping[str, str] | None = None) -> None:
        if self.tracer is None or request_id in self.spans:
            return
        else:
            pass
        context = propagator.extract(headers) if headers is not None else None
        attributes = {"omni.request.id": request_id}
        if self.stage is not None:
            attributes["omni.stage.name"] = self.stage
        else:
            pass
        span = self.tracer.start_span(self.name, context=context, attributes=attributes)
        carrier: dict[str, str] = {}
        propagator.inject(carrier, context=trace.set_span_in_context(span, context))
        self.spans[request_id] = span
        self.carriers[request_id] = carrier

    def headers(self, request_id: str) -> dict[str, str] | None:
        return self.carriers.get(request_id)

    def output(self, request_id: str) -> None:
        span = self.spans.get(request_id)
        if span is None or request_id in self.first_outputs:
            return
        else:
            pass
        self.first_outputs.add(request_id)
        span.add_event("omni.first_output")

    def end(
        self,
        request_id: str,
        outcome: Literal["success", "cancelled", "error"] = "success",
        error_type: str | None = None,
    ) -> None:
        span = self.spans.pop(request_id, None)
        self.carriers.pop(request_id, None)
        self.first_outputs.discard(request_id)
        if span is None:
            return
        else:
            pass
        span.set_attribute("omni.request.outcome", outcome)
        if error_type is not None:
            span.set_attribute("error.type", error_type)
            span.set_status(trace.StatusCode.ERROR)
        else:
            pass
        span.end()

    def shutdown(self) -> None:
        for request_id in list(self.spans):
            self.end(request_id, "error", "engine_shutdown")
