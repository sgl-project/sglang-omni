# SPDX-License-Identifier: Apache-2.0
"""Session runtime contracts that depend on scheduling the wire cannot pin."""

from __future__ import annotations

import asyncio
import logging

import pytest

from sglang_omni.serve.realtime.control import Closed, Drained, Failure, UnitCompleted
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    OutputEvent,
    ResponseFinished,
    ResponseStarted,
)
from sglang_omni.serve.realtime.output_buffer import OutputBuffer
from sglang_omni.serve.realtime.runtime import SessionRuntime
from sglang_omni.serve.realtime.schema import SessionConfiguration
from sglang_omni.serve.realtime.types import (
    Capabilities,
    Envelope,
    InteractionAdapter,
    OutputSink,
    RuntimeLimits,
    Unit,
)

MODEL_NAME = "duplex-test"
SAMPLE_RATE = 16000
NATIVE_UNIT_MS = 20
UNIT_BYTES = SAMPLE_RATE * NATIVE_UNIT_MS // 1000 * 2
RUNTIME_LOGGER_NAME = "sglang_omni.serve.realtime.runtime"


class GatedAdapter(InteractionAdapter):
    """Blocks on the first unit until released, emitting a fixed reply per unit."""

    def __init__(self, reply: list[OutputEvent]) -> None:
        self.reply = reply
        self.units: list[Unit] = []
        self.output_sink: OutputSink | None = None
        self.has_started = asyncio.Event()
        self.release = asyncio.Event()

    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        self.output_sink = emit

    async def process(self, unit: Unit) -> int:
        assert self.output_sink is not None
        self.units.append(unit)
        for event in self.reply:
            await self.output_sink(event)
        self.has_started.set()
        await self.release.wait()
        return unit.real_samples

    async def close(self) -> None:
        pass


class GatedPrefetchAdapter(GatedAdapter):
    def __init__(self) -> None:
        super().__init__([])
        self.prefetch_started = asyncio.Event()
        self.prefetch_release = asyncio.Event()
        self.prefetch_finished = asyncio.Event()
        self.prefetch_cancelled = asyncio.Event()
        self.clear_started = asyncio.Event()
        self.prefetched: list[tuple[int, float, bytes]] = []
        self.calls: list[str] = []

    async def prefetch_image(self, unit_index: int, t_ms: float, image: bytes) -> None:
        self.prefetch_started.set()
        try:
            await self.prefetch_release.wait()
        except asyncio.CancelledError:
            self.prefetch_cancelled.set()
            raise
        self.prefetched.append((unit_index, t_ms, image))
        self.calls.append("prefetch")
        self.prefetch_finished.set()

    async def clear(self) -> int:
        self.calls.append("clear")
        self.clear_started.set()
        return 0

    async def close(self) -> None:
        self.calls.append("close")


async def open_runtime(adapter: GatedAdapter) -> SessionRuntime:
    runtime = SessionRuntime(
        MODEL_NAME, Capabilities(), lambda: adapter, RuntimeLimits()
    )
    await runtime.update({}, "client_update")
    return runtime


async def receive_until(
    runtime: SessionRuntime, event_type: type[Drained | UnitCompleted | Closed]
) -> list[Envelope]:
    envelopes: list[Envelope] = []
    async for envelope in runtime.outputs():
        envelopes.append(envelope)
        if isinstance(envelope.event, event_type):
            break
        else:
            pass
    return envelopes


@pytest.mark.asyncio
async def test_eos_marks_only_the_last_unit_when_end_arrives_mid_backlog() -> None:
    adapter = GatedAdapter([])
    runtime = await open_runtime(adapter)
    await runtime.append(b"\1" * UNIT_BYTES, 0, None, "client_append_0")
    await adapter.has_started.wait()
    await runtime.append(b"\1" * UNIT_BYTES * 2, 1, None, "client_append_1")
    await runtime.end("client_end")

    adapter.release.set()
    drained = (await receive_until(runtime, Drained))[-1].event
    await runtime.close("client_closed")
    assert [unit.eos for unit in adapter.units] == [False, False, True]
    assert isinstance(drained, Drained)
    assert (drained.accepted_end_ms, drained.consumed_ms) == (60.0, 60.0)


@pytest.mark.asyncio
async def test_image_append_schedules_prefetch_without_waiting() -> None:
    adapter = GatedPrefetchAdapter()
    runtime = await open_image_runtime(adapter)
    image = b"\xff\xd8frame"

    await asyncio.wait_for(runtime.append_image(image, 0.0, "frame"), 1)
    assert runtime.pending_frames == {0: [(0.0, image)]}
    await asyncio.wait_for(adapter.prefetch_started.wait(), 1)
    assert not adapter.prefetch_finished.is_set()

    adapter.prefetch_release.set()
    await asyncio.wait_for(adapter.prefetch_finished.wait(), 1)
    assert adapter.prefetched == [(0, 0.0, image)]
    await runtime.close("client_closed")


async def open_image_runtime(adapter: GatedPrefetchAdapter) -> SessionRuntime:
    runtime = SessionRuntime(
        MODEL_NAME,
        Capabilities(input_modalities=("audio", "image")),
        lambda: adapter,
        RuntimeLimits(),
    )
    await runtime.update({}, "client_update")
    return runtime


@pytest.mark.asyncio
async def test_clear_waits_for_inflight_image_prefetch() -> None:
    adapter = GatedPrefetchAdapter()
    runtime = await open_image_runtime(adapter)
    await runtime.append_image(b"\xff\xd8frame", 0.0, "frame")
    await asyncio.wait_for(adapter.prefetch_started.wait(), 1)

    clear_task = asyncio.create_task(runtime.clear("client_clear"))
    await asyncio.sleep(0)
    assert not clear_task.done()
    assert not adapter.clear_started.is_set()

    adapter.prefetch_release.set()
    await asyncio.wait_for(clear_task, 1)
    assert runtime.pending_frames == {}
    assert adapter.calls == ["prefetch", "clear"]

    await runtime.close("client_closed")


@pytest.mark.asyncio
async def test_close_cancels_inflight_image_prefetch() -> None:
    adapter = GatedPrefetchAdapter()
    runtime = await open_image_runtime(adapter)
    await runtime.append_image(b"\xff\xd8frame", 0.0, "frame")
    await asyncio.wait_for(adapter.prefetch_started.wait(), 1)

    await runtime.close("client_closed")

    assert adapter.prefetch_cancelled.is_set()
    assert runtime.image_prefetch_tasks == set()
    assert runtime.state == "CLOSED"
    assert adapter.calls == ["close"]


@pytest.mark.asyncio
async def test_close_finishes_only_responses_the_client_has_seen() -> None:
    adapter = GatedAdapter([ResponseStarted("seen"), ResponseStarted("unseen")])
    runtime = await open_runtime(adapter)
    await runtime.append(b"\1" * UNIT_BYTES, 0, None, "client_append_0")
    await adapter.has_started.wait()
    adapter.release.set()
    for envelope in await receive_until(runtime, UnitCompleted):
        if envelope.event == ResponseStarted("seen"):
            runtime.output_buffer.before_send(envelope)
            runtime.output_buffer.sent(envelope)
        else:
            pass

    await runtime.close("client_closed")
    closing = [envelope.event for envelope in await receive_until(runtime, Closed)]

    finished = [event for event in closing if isinstance(event, ResponseFinished)]
    assert [event.response_id for event in finished] == ["seen"]
    assert finished[0].status == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_message", "failure_code", "has_traceback"),
    [
        (
            "context_exhausted: the session reached the thinker context length",
            "context_exhausted",
            False,
        ),
        ("talker step failed", "internal", True),
    ],
)
async def test_unit_failure_closes_session_and_logs_once(
    failure_message: str,
    failure_code: str,
    has_traceback: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingAdapter(GatedAdapter):
        async def process(self, unit: Unit) -> int:
            raise RuntimeError(failure_message)

    with caplog.at_level(logging.WARNING, logger=RUNTIME_LOGGER_NAME):
        runtime = await open_runtime(FailingAdapter([]))
        await runtime.append(b"\1" * UNIT_BYTES, 0, None, "append")
        envelopes = await asyncio.wait_for(receive_until(runtime, Closed), 5)
    failures = [entry.event for entry in envelopes if isinstance(entry.event, Failure)]
    assert len(failures) == 1
    assert failures[0].code == failure_code
    assert failures[0].is_fatal
    assert failure_message in failures[0].message
    runtime_records = [
        record for record in caplog.records if record.name == RUNTIME_LOGGER_NAME
    ]
    assert len(runtime_records) == 1
    assert runtime_records[0].levelno == logging.ERROR
    assert (runtime_records[0].exc_info is not None) == has_traceback


def test_output_budget_counts_outbound_events_only() -> None:
    buffer = OutputBuffer(RuntimeLimits(max_output_bytes=1024, max_output_events=2))
    unit = Unit(0, 0, bytes(32000), 16000, images=(bytes(512 * 1024),) * 4)
    completed = Envelope(event=UnitCompleted(unit.unit_id), unit=unit)
    buffer.enqueue(completed)
    with pytest.raises(RuntimeError, match="outbound event budget exhausted"):
        buffer.enqueue(Envelope(event=AudioDelta("response", "item", bytes(2048))))
    buffer.enqueue(completed)
    with pytest.raises(RuntimeError, match="outbound event budget exhausted"):
        buffer.enqueue(completed)
    assert [buffer.dequeue(), buffer.dequeue(), buffer.dequeue()] == [
        completed,
        completed,
        None,
    ]
