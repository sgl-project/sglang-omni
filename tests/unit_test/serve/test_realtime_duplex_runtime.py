# SPDX-License-Identifier: Apache-2.0
"""Session runtime contracts that depend on scheduling the wire cannot pin."""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path

import pytest

import sglang_omni.serve.realtime.runtime as runtime_module
from sglang_omni.profiler.duplex_events import SessionUnitReadyObservation
from sglang_omni.profiler.event_recorder import get_recorder
from sglang_omni.serve.realtime.control import Closed, Drained, Failure, UnitCompleted
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    OutputEvent,
    ResponseFinished,
    ResponseStarted,
)
from sglang_omni.serve.realtime.output_buffer import (
    ENVELOPE_BYTES,
    OutputBuffer,
    envelope_size_bytes,
)
from sglang_omni.serve.realtime.runtime import SessionRuntime
from sglang_omni.serve.realtime.schema import SessionConfiguration
from sglang_omni.serve.realtime.types import (
    Capabilities,
    Envelope,
    InteractionAdapter,
    OutputSink,
    ProtocolError,
    RuntimeLimits,
    Unit,
)

MODEL_NAME = "duplex-test"
SAMPLE_RATE = 16000
NATIVE_UNIT_MS = 20
UNIT_BYTES = SAMPLE_RATE * NATIVE_UNIT_MS // 1000 * 2
RUNTIME_LOGGER_NAME = "sglang_omni.serve.realtime.runtime"
UPDATE_DEADLINE_S = 600.0
TEST_WAIT_TIMEOUT_SECONDS = 5.0


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


class SlowAdmissionAdapter(GatedAdapter):
    """Reports when admission starts and holds it until released."""

    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        self.has_started.set()
        await self.release.wait()
        await super().open(session_id, config, emit)


@pytest.fixture
def deadline_reached(monkeypatch: pytest.MonkeyPatch) -> asyncio.Event:
    """Holds the update deadline's sleep until the test sets the returned event."""
    is_reached = asyncio.Event()
    real_sleep = asyncio.sleep

    async def sleep_until_reached(delay_s: float) -> None:
        if delay_s == UPDATE_DEADLINE_S:
            await is_reached.wait()
        else:
            await real_sleep(delay_s)

    monkeypatch.setattr(asyncio, "sleep", sleep_until_reached)
    return is_reached


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


def ready_observation(
    timestamp_ns: int, run_id: str = "run-a"
) -> SessionUnitReadyObservation:
    return SessionUnitReadyObservation(timestamp_ns=timestamp_ns, run_id=run_id)


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
async def test_ready_before_profiler_start_is_not_retrofitted(tmp_path: Path) -> None:
    recorder = get_recorder()
    if recorder.is_active():
        recorder.stop()
    else:
        pass
    adapter = GatedAdapter([])
    runtime = await open_runtime(adapter)
    await runtime.append(b"\1" * UNIT_BYTES * 2, 0, None, "client_append_0")
    await adapter.has_started.wait()

    recorder.start("run-a", str(tmp_path), "coordinator")
    await runtime.end("client_end")
    adapter.release.set()
    await receive_until(runtime, Drained)
    await runtime.close("client_closed")
    recorder.stop()

    assert [unit.ready_observation for unit in adapter.units] == [None, None]


@pytest.mark.asyncio
async def test_multiple_units_from_one_append_share_ready_timestamp(
    monkeypatch,
) -> None:
    ready_observations = iter((ready_observation(100), ready_observation(200)))
    monkeypatch.setattr(
        runtime_module,
        "capture_session_unit_ready",
        lambda: next(ready_observations),
    )
    adapter = GatedAdapter([])
    runtime = await open_runtime(adapter)
    await runtime.append(b"\1" * UNIT_BYTES * 2, 0, None, "client_append_0")
    await adapter.has_started.wait()
    await runtime.end("client_end")

    adapter.release.set()
    await receive_until(runtime, Drained)
    await runtime.close("client_closed")

    assert [unit.ready_observation.timestamp_ns for unit in adapter.units] == [100, 100]


@pytest.mark.parametrize("has_empty_tail", [False, True])
@pytest.mark.asyncio
async def test_eos_marks_terminal_units_ready_at_eos(
    monkeypatch: pytest.MonkeyPatch, has_empty_tail: bool
) -> None:
    ready_observations = iter((ready_observation(100), ready_observation(200)))
    monkeypatch.setattr(
        runtime_module,
        "capture_session_unit_ready",
        lambda: next(ready_observations),
    )
    adapter = GatedAdapter([])
    runtime = await open_runtime(adapter)
    append_bytes = UNIT_BYTES if has_empty_tail else UNIT_BYTES + UNIT_BYTES // 2
    await runtime.append(b"\1" * append_bytes, 0, None, "client_append_0")
    await adapter.has_started.wait()
    await runtime.end("client_end")

    adapter.release.set()
    await receive_until(runtime, Drained)
    await runtime.close("client_closed")

    assert [unit.ready_observation.timestamp_ns for unit in adapter.units] == [100, 200]
    if has_empty_tail:
        assert [unit.real_samples for unit in adapter.units] == [
            UNIT_BYTES // 2,
            0,
        ]
    else:
        pass


@pytest.mark.asyncio
async def test_optional_image_does_not_delay_audio_ready_timestamp(
    monkeypatch,
) -> None:
    ready_observations = iter(
        (ready_observation(100), ready_observation(200), ready_observation(300))
    )
    monkeypatch.setattr(
        runtime_module,
        "capture_session_unit_ready",
        lambda: next(ready_observations),
    )
    adapter = GatedAdapter([])
    runtime = SessionRuntime(
        MODEL_NAME,
        Capabilities(input_modalities=("audio", "image")),
        lambda: adapter,
        RuntimeLimits(),
    )
    await runtime.update({}, "client_update")
    await runtime.append(b"\1" * UNIT_BYTES, 0, None, "client_append_0")
    await adapter.has_started.wait()
    await runtime.append(b"\1" * UNIT_BYTES, 1, None, "client_append_1")
    await runtime.append_image(b"\xff\xd8frame", 20, "client_image")
    await runtime.end("client_end")
    adapter.release.set()
    await receive_until(runtime, Drained)

    await runtime.close("client_closed")

    assert adapter.units[1].images == (b"\xff\xd8frame",)
    assert [unit.ready_observation.timestamp_ns for unit in adapter.units] == [100, 200]


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


@pytest.mark.asyncio
async def test_update_deadline_does_not_cut_off_an_admission_in_progress(
    deadline_reached: asyncio.Event,
) -> None:
    adapter = SlowAdmissionAdapter([])
    runtime = SessionRuntime(
        MODEL_NAME,
        Capabilities(),
        lambda: adapter,
        RuntimeLimits(session_update_timeout_s=UPDATE_DEADLINE_S),
    )
    runtime.notify_created()
    update_task = asyncio.create_task(runtime.update({}, "update"))
    await adapter.has_started.wait()
    deadline_reached.set()
    await asyncio.sleep(0)
    adapter.release.set()
    await asyncio.wait_for(update_task, 5)
    await asyncio.sleep(0)
    await runtime.close("client_closed")
    envelopes = await asyncio.wait_for(receive_until(runtime, Closed), 5)

    assert [
        entry.event for entry in envelopes if isinstance(entry.event, Failure)
    ] == []
    assert envelopes[-1].event == Closed("client_closed")


@pytest.mark.asyncio
async def test_update_after_the_deadline_is_rejected_without_admission(
    deadline_reached: asyncio.Event,
) -> None:
    adapter = SlowAdmissionAdapter([])
    adapter.release.set()
    runtime = SessionRuntime(
        MODEL_NAME,
        Capabilities(),
        lambda: adapter,
        RuntimeLimits(session_update_timeout_s=UPDATE_DEADLINE_S),
    )
    runtime.notify_created()
    deadline_reached.set()
    await asyncio.sleep(0)

    with pytest.raises(ProtocolError, match="closing"):
        await runtime.update({}, "update")
    envelopes = await asyncio.wait_for(receive_until(runtime, Closed), 5)

    assert not adapter.has_started.is_set()
    failures = [entry.event for entry in envelopes if isinstance(entry.event, Failure)]
    assert [failure.code for failure in failures] == ["session_update_timeout"]


def test_an_audio_delta_is_sized_by_its_base64_audio_and_the_envelope() -> None:
    pcm = bytes(range(256)) * 7
    envelope = Envelope(event=AudioDelta("response", "item", pcm))
    assert envelope_size_bytes(envelope) == len(base64.b64encode(pcm)) + ENVELOPE_BYTES


def test_output_budget_counts_outbound_events_only() -> None:
    buffer = OutputBuffer(RuntimeLimits(max_output_bytes=3 * ENVELOPE_BYTES))
    unit = Unit(0, 0, bytes(32000), 16000, images=(bytes(512 * 1024),) * 4)
    completed = Envelope(event=UnitCompleted(unit.unit_id), unit=unit)
    buffer.enqueue(completed)
    with pytest.raises(RuntimeError, match="outbound event budget exhausted"):
        buffer.enqueue(Envelope(event=AudioDelta("response", "item", bytes(2048))))
    buffer.enqueue(completed)
    buffer.enqueue(completed)
    with pytest.raises(RuntimeError, match="outbound event budget exhausted"):
        buffer.enqueue(completed)
    assert buffer.queued_bytes == 3 * ENVELOPE_BYTES
    assert [buffer.dequeue() for _ in range(4)] == [completed] * 3 + [None]


@pytest.mark.asyncio
async def test_competing_closes_keep_initial_failure_reason() -> None:
    """Concurrent close requests must preserve the initial failure reported to clients."""
    runtime = await open_runtime(GatedAdapter([]))
    try:
        runtime.fail("context full", "context_exhausted")
        await asyncio.wait_for(
            asyncio.gather(runtime.close("disconnect"), runtime.close("client_closed")),
            TEST_WAIT_TIMEOUT_SECONDS,
        )
    finally:
        await asyncio.wait_for(runtime.close("disconnect"), TEST_WAIT_TIMEOUT_SECONDS)

    assert runtime.close_reason == "context_exhausted"
