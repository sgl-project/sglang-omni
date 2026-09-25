# SPDX-License-Identifier: Apache-2.0

import asyncio
import queue
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import httpx
import pytest
import torch
from opentelemetry.sdk.trace import Tracer, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from sglang_omni import tracing
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.local_dispatch import LocalStageDispatcher
from sglang_omni.proto import (
    CompleteMessage,
    DataReadyMessage,
    StagePayload,
    SubmitMessage,
)
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from tests.unit_test.fixtures.pipeline_fakes import make_stage_payload
from tests.unit_test.pipeline.helpers import make_stage

HEADERS = {
    "traceparent": "00-12345678901234567890123456789012-1234567890123456-01",
    "tracestate": "vendor=state",
    "baggage": "tenant=gold",
}


@pytest.fixture
def recording() -> Iterator[tuple[Tracer, InMemorySpanExporter]]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider.get_tracer("test"), exporter
    provider.shutdown()


def test_control_message_round_trip_and_unsampled(
    recording: tuple[Tracer, InMemorySpanExporter],
) -> None:
    tracer, exporter = recording
    headers = dict(HEADERS, traceparent=HEADERS["traceparent"][:-2] + "00")
    traces = tracing.RequestTraces(tracer, "omni.pipeline")
    traces.start("req", headers)
    carrier = traces.headers("req")
    for msg in [
        SubmitMessage("req", {}, trace_headers=carrier),
        DataReadyMessage("req", "a", "b", {}, trace_headers=carrier),
    ]:
        assert type(msg).from_dict(msg.to_dict()).trace_headers == carrier
    assert carrier["tracestate"] == "vendor=state"
    assert carrier["baggage"] == "tenant=gold"
    assert carrier["traceparent"].endswith("-00")
    traces.end("req")
    assert not exporter.get_finished_spans()
    assert tracing.create_tracer_provider(None) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "cancelled", "error"])
async def test_local_stages_share_parent_and_close(
    recording: tuple[Tracer, InMemorySpanExporter],
    outcome: Literal["success", "cancelled", "error"],
) -> None:
    tracer, exporter = recording
    dispatcher = LocalStageDispatcher()
    source = make_stage(
        name="source",
        get_next=lambda request_id, result: "target",
        tracer=tracer,
        local_dispatcher=dispatcher,
        same_process_targets={"target"},
        endpoints={"target": "local"},
    )
    target = make_stage(name="target", tracer=tracer, local_dispatcher=dispatcher)
    dispatcher.register_many([source, target])
    pipeline = tracing.RequestTraces(tracer, "omni.pipeline")
    pipeline.start("req", HEADERS)
    payload = make_stage_payload(request_id="req")
    await source.on_submit(
        SubmitMessage("req", payload, trace_headers=pipeline.headers("req"))
    )
    source.traces.output("req")
    source.traces.output("req")
    await source.route_result("req", payload)
    if outcome == "error":
        await target.send_failure("req", "private error details")
    elif outcome == "cancelled":
        target.on_abort("req")
        await target.on_submit(SubmitMessage("req", payload, trace_headers=HEADERS))
    else:
        await target.route_result("req", payload)
    pipeline.end("req", outcome, "stage_error" if outcome == "error" else None)
    spans = exporter.get_finished_spans()
    assert len(spans) == 3
    source_span, target_span, pipeline_span = spans
    assert source_span.parent.span_id == pipeline_span.context.span_id
    assert target_span.parent.span_id == source_span.context.span_id
    assert target_span.attributes["omni.request.outcome"] == outcome
    assert len(source_span.events) == 1
    assert not target.traces.spans
    assert all("private" not in str(s.attributes) for s in spans)


@pytest.mark.asyncio
async def test_coordinator_waits_for_all_terminal_stages(
    recording: tuple[Tracer, InMemorySpanExporter],
) -> None:
    tracer, exporter = recording
    coordinator = Coordinator(
        "ipc:///tmp/unused-trace-complete",
        "ipc:///tmp/unused-trace-abort",
        "entry",
        terminal_stages=["audio", "text"],
        tracer=tracer,
    )
    coordinator.register_stage("entry", "ipc:///tmp/unused-trace-entry")
    submitted = []

    async def submit(stage: str, endpoint: str, msg: SubmitMessage) -> None:
        submitted.append(msg)

    coordinator.control_plane.submit_to_stage = submit
    await coordinator.submit_request(
        "req", {"private": "prompt"}, trace_headers=HEADERS
    )
    assert submitted[0].trace_headers["tracestate"] == "vendor=state"
    await coordinator.handle_completion(CompleteMessage("req", "text", True, "text"))
    assert not exporter.get_finished_spans()
    await coordinator.handle_completion(CompleteMessage("req", "audio", True, "audio"))
    assert len(exporter.get_finished_spans()) == 1
    assert not coordinator.traces.spans
    coordinator.control_plane.close()


class TraceProbeScheduler:
    def __init__(self, *, emit_stream: bool = False) -> None:
        self.inbox: queue.Queue[IncomingMessage] = queue.Queue()
        self.outbox: queue.Queue[OutgoingMessage] = queue.Queue()
        self.requires_tp_work_fanout = True
        self.emit_stream = emit_stream
        self.running = False

    def start(self) -> None:
        self.running = True
        while self.running:
            try:
                message = self.inbox.get(timeout=0.1)
            except queue.Empty:
                continue
            if message.type == "new_request":
                assert isinstance(message.data, StagePayload)
                message.data.data["text"] = "hello"
                if self.emit_stream:
                    self.outbox.put(
                        OutgoingMessage(
                            request_id=message.request_id,
                            type="stream",
                            data=torch.ones(1),
                        )
                    )
                self.outbox.put(
                    OutgoingMessage(
                        request_id=message.request_id, type="result", data=message.data
                    )
                )

    def stop(self) -> None:
        self.running = False

    def abort(self, request_id: str) -> None:
        pass


def make_trace_probe_scheduler(*, emit_stream: bool = False) -> TraceProbeScheduler:
    return TraceProbeScheduler(emit_stream=emit_stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("colocated", [False, True])
async def test_process_pipeline_exports_connected_spans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, colocated: bool
) -> None:
    import http.server
    import threading

    instrumentation = pytest.importorskip("opentelemetry.instrumentation.fastapi")
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")

    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    from sglang_omni.client import Client
    from sglang_omni.config import (
        EndpointsConfig,
        FactoryArgs,
        PipelineConfig,
        ProcessConfig,
        StageConfig,
    )
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
    from sglang_omni.serve.openai_api import create_app

    exported = []

    class Collector(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            batch = ExportTraceServiceRequest.FromString(
                self.rfile.read(int(self.headers["Content-Length"]))
            )
            for resource in batch.resource_spans:
                for scope in resource.scope_spans:
                    exported.extend(scope.spans)
            self.send_response(200)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    collector = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Collector)
    thread = threading.Thread(target=collector.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf")
    target_process = "pair" if colocated else "target"
    config = PipelineConfig(
        model_path="model",
        entry_stage="producer",
        stages=[
            StageConfig(
                name="producer",
                process="pair",
                factory_path="tests.unit_test.pipeline.test_tracing.make_trace_probe_scheduler",
                factory=FactoryArgs(emit_stream=True),
                next="consumer",
                stream_to=["consumer"],
            ),
            StageConfig(
                name="consumer",
                process=target_process,
                factory_path="tests.unit_test.pipeline.test_tracing.make_trace_probe_scheduler",
                terminal=True,
                can_accept_stream_before_payload=True,
            ),
        ],
        processes={name: ProcessConfig() for name in {"pair", target_process}},
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
    )
    runner = MultiProcessPipelineRunner(
        config,
        otlp_traces_endpoint=f"http://127.0.0.1:{collector.server_port}/v1/traces",
    )
    try:
        await runner.start(timeout=30)
        app = create_app(Client(runner.coordinator), model_name="trace-test")
        instrumentation.FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=runner.tracer_provider,
            exclude_spans=["send", "receive"],
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            for sampled in ("01", "00"):
                headers = dict(
                    HEADERS, traceparent=HEADERS["traceparent"][:-2] + sampled
                )
                response = await asyncio.wait_for(
                    client.post(
                        "/v1/chat/completions",
                        headers=headers,
                        json={
                            "model": "trace-test",
                            "messages": [{"role": "user", "content": "hi"}],
                        },
                    ),
                    timeout=10,
                )
                assert response.status_code == 200, response.text
                assert response.json()["choices"][0]["message"]["content"] == "hello"
    finally:
        await runner.stop()
        collector.shutdown()
        collector.server_close()
        thread.join()
    assert len(exported) == 4
    server = next(span for span in exported if span.kind == 2)
    pipeline = next(span for span in exported if span.name == "omni.pipeline")
    stages = [span for span in exported if span.name == "omni.stage"]
    source = next(span for span in stages if span.parent_span_id == pipeline.span_id)
    target = next(span for span in stages if span.parent_span_id == source.span_id)
    assert all(
        span.trace_id.hex() == "12345678901234567890123456789012" for span in exported
    )
    assert server.parent_span_id.hex() == "1234567890123456"
    assert pipeline.parent_span_id == server.span_id
    assert target.end_time_unix_nano >= target.start_time_unix_nano
    assert all(span.trace_state == "vendor=state" for span in exported)
