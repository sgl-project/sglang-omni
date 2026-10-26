# SPDX-License-Identifier: Apache-2.0
"""Session runtime contracts that depend on scheduling the wire cannot pin."""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pytest

import sglang_omni.serve.realtime.runtime as runtime_module
from sglang_omni.profiler.duplex_events import SessionUnitReadyObservation
from sglang_omni.profiler.event_recorder import get_recorder
from sglang_omni.serve.realtime.control import (
    Closed,
    Drained,
    Failure,
    UnitCompleted,
    Updated,
)
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    AudioFinished,
    OutputEvent,
    ResponseFinished,
    ResponseProgress,
    ResponseStarted,
    TextDelta,
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
CANCELLED_RESPONSE_ID = "resp_1"
CANCELLED_ITEM_ID = "item_1"
TEST_WAIT_TIMEOUT_SECONDS = 5.0
ADMISSION_CLEANUP_TIMEOUT_SECONDS = 0.5


class GatedAdapter(InteractionAdapter):
    """Blocks on the first unit until released, emitting a fixed reply per unit."""

    def __init__(self, reply: list[OutputEvent]) -> None:
        self.reply = reply
        self.units: list[Unit] = []
        self.output_sink: OutputSink | None = None
        self.has_started = asyncio.Event()
        self.release = asyncio.Event()
        self.close_call_count = 0

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
        self.close_call_count += 1


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


@asynccontextmanager
async def running_runtime(runtime: SessionRuntime) -> AsyncIterator[SessionRuntime]:
    runtime.notify_created()
    try:
        yield runtime
    finally:
        await asyncio.wait_for(
            runtime.close("client_closed"), TEST_WAIT_TIMEOUT_SECONDS
        )


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


def test_a_cancelled_response_drops_only_its_queued_audio() -> None:
    """A socket slower than the reply still held two audio deltas of a cancelled reply, and both left after the cancel. They are dropped at the cancelled response.done, while the reply's text, progress and audio done, an earlier reply's queued audio and the terminals stay in order with the byte count exact."""
    buffer = OutputBuffer(RuntimeLimits())
    earlier_audio = AudioDelta("resp_0", "item_0", bytes(UNIT_BYTES))
    earlier_finished = ResponseFinished(
        "resp_0", "item_0", "earlier", True, "completed", "stop"
    )
    cancelled_text = TextDelta(CANCELLED_RESPONSE_ID, CANCELLED_ITEM_ID, "unheard")
    cancelled_progress = ResponseProgress(
        CANCELLED_RESPONSE_ID, CANCELLED_ITEM_ID, 7, 40.0
    )
    cancelled_audio_done = AudioFinished(CANCELLED_RESPONSE_ID, CANCELLED_ITEM_ID)
    cancelled_finished = ResponseFinished(
        CANCELLED_RESPONSE_ID,
        CANCELLED_ITEM_ID,
        "unheard",
        True,
        "cancelled",
        "interrupted",
    )
    for event in (
        ResponseStarted("resp_0"),
        earlier_audio,
        earlier_finished,
        ResponseStarted(CANCELLED_RESPONSE_ID),
        AudioDelta(CANCELLED_RESPONSE_ID, CANCELLED_ITEM_ID, bytes(UNIT_BYTES)),
        cancelled_text,
        AudioDelta(CANCELLED_RESPONSE_ID, CANCELLED_ITEM_ID, bytes(UNIT_BYTES)),
        cancelled_progress,
        cancelled_audio_done,
        cancelled_finished,
    ):
        buffer.emit(event, None, ("text", "audio"))

    assert buffer.queued_bytes == 256 + 1112 + 263 + 256 + 263 + 256 + 256 + 263
    drained_events: list[OutputEvent] = []
    while (envelope := buffer.dequeue()) is not None:
        drained_events.append(envelope.event)
    assert drained_events == [
        ResponseStarted("resp_0"),
        earlier_audio,
        earlier_finished,
        ResponseStarted(CANCELLED_RESPONSE_ID),
        cancelled_text,
        cancelled_progress,
        cancelled_audio_done,
        cancelled_finished,
    ]


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


@pytest.mark.parametrize(
    "termination", ["cancelled_future", "cancelled_task", "client_close"]
)
@pytest.mark.asyncio
async def test_process_cancellation_releases_session_resources(
    termination: Literal["cancelled_future", "cancelled_task", "client_close"],
) -> None:
    """Cancelling input processing must close the session and release its resources."""

    class CancellableAdapter(GatedAdapter):
        async def process(self, unit: Unit) -> int:
            self.has_started.set()
            if termination == "cancelled_future":
                consumed_samples: asyncio.Future[int] = (
                    asyncio.get_running_loop().create_future()
                )
                consumed_samples.cancel()
                return await consumed_samples
            else:
                await self.release.wait()
                return unit.real_samples

    adapter = CancellableAdapter([])
    runtime = SessionRuntime(
        MODEL_NAME, Capabilities(), lambda: adapter, RuntimeLimits()
    )
    async with running_runtime(runtime):
        await runtime.update({}, "client_update")
        await runtime.append(b"\1" * UNIT_BYTES, 0, None, "append")
        await asyncio.wait_for(adapter.has_started.wait(), TEST_WAIT_TIMEOUT_SECONDS)
        if termination == "cancelled_task":
            assert runtime.pump_task is not None
            runtime.pump_task.cancel()
        elif termination == "client_close":
            await runtime.close("client_closed")
        else:
            pass

        envelopes = await asyncio.wait_for(
            receive_until(runtime, Closed), TEST_WAIT_TIMEOUT_SECONDS
        )
        if termination == "client_close":
            assert envelopes[-1].event == Closed("client_closed")
            assert not any(
                isinstance(envelope.event, Failure) for envelope in envelopes
            )
        else:
            assert closing_failure(envelopes) == ("internal", "internal")
        assert adapter.close_call_count == 1
        assert runtime.state == "CLOSED"


@pytest.mark.asyncio
async def test_admission_finishing_after_close_releases_adapter_without_opening() -> (
    None
):
    """An admission that succeeds after the session started closing must release the adapter and never open the session."""

    adapter = SlowAdmissionAdapter([])
    runtime = SessionRuntime(
        MODEL_NAME, Capabilities(), lambda: adapter, RuntimeLimits()
    )
    runtime.notify_created()
    update_task = asyncio.create_task(runtime.update({}, "open"))
    await asyncio.wait_for(adapter.has_started.wait(), TEST_WAIT_TIMEOUT_SECONDS)
    close_task = asyncio.create_task(runtime.close("client_closed"))
    await asyncio.sleep(0)
    adapter.release.set()
    await asyncio.wait_for(update_task, TEST_WAIT_TIMEOUT_SECONDS)
    await asyncio.wait_for(close_task, TEST_WAIT_TIMEOUT_SECONDS)

    envelopes = await asyncio.wait_for(
        receive_until(runtime, Closed), TEST_WAIT_TIMEOUT_SECONDS
    )
    assert not any(isinstance(envelope.event, Updated) for envelope in envelopes)
    assert adapter.close_call_count == 1
    assert runtime.pump_task is None
    assert runtime.state == "CLOSED"


@pytest.mark.parametrize("cancellation_stage", ["open", "admission_cleanup"])
@pytest.mark.parametrize("should_close_first", [False, True])
@pytest.mark.asyncio
async def test_cancelled_admission_closes_resources_and_rejects_queued_update(
    cancellation_stage: Literal["open", "admission_cleanup"],
    should_close_first: bool,
) -> None:
    """Cancelling admission must release resources and reject queued session updates."""

    class CancellableAdmissionAdapter(GatedAdapter):
        def __init__(self) -> None:
            super().__init__([])
            self.cleanup_started: asyncio.Event = asyncio.Event()
            self.is_resource_open: bool = False

        async def open(
            self, session_id: str, config: SessionConfiguration, emit: OutputSink
        ) -> None:
            self.is_resource_open = True
            self.has_started.set()
            if cancellation_stage == "admission_cleanup":
                raise RuntimeError("pipeline unavailable")
            else:
                await self.release.wait()

        async def close(self) -> None:
            await super().close()
            if cancellation_stage == "admission_cleanup" and self.close_call_count == 1:
                self.cleanup_started.set()
                await self.release.wait()
            else:
                pass
            self.is_resource_open = False

    adapter = CancellableAdmissionAdapter()
    factory_call_count = 0

    def create_adapter() -> InteractionAdapter:
        nonlocal factory_call_count
        factory_call_count += 1
        return adapter

    runtime = SessionRuntime(
        MODEL_NAME, Capabilities(), create_adapter, RuntimeLimits()
    )
    async with running_runtime(runtime):
        update_task = asyncio.create_task(runtime.update({}, "open"))
        queued_update_task: asyncio.Task[None] | None = None
        client_close_task: asyncio.Task[None] | None = None
        try:
            cancellation_started = (
                adapter.cleanup_started
                if cancellation_stage == "admission_cleanup"
                else adapter.has_started
            )
            await asyncio.wait_for(
                cancellation_started.wait(), TEST_WAIT_TIMEOUT_SECONDS
            )
            queued_update_task = asyncio.create_task(runtime.update({}, "queued"))
            await asyncio.sleep(0)
            if should_close_first:
                client_close_task = asyncio.create_task(runtime.close("client_closed"))
                await asyncio.sleep(0)
            else:
                pass
            update_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(update_task, TEST_WAIT_TIMEOUT_SECONDS)
            with pytest.raises(ProtocolError, match="closing"):
                await asyncio.wait_for(queued_update_task, TEST_WAIT_TIMEOUT_SECONDS)

            envelopes = await asyncio.wait_for(
                receive_until(runtime, Closed), TEST_WAIT_TIMEOUT_SECONDS
            )
            if should_close_first:
                assert envelopes[-1].event == Closed("client_closed")
                assert not any(
                    isinstance(envelope.event, Failure) for envelope in envelopes
                )
            else:
                assert closing_failure(envelopes) == ("internal", "internal")
                assert [
                    envelope.event.client_event_id
                    for envelope in envelopes
                    if isinstance(envelope.event, Failure)
                ] == ["open"]
            assert factory_call_count == 1
            assert adapter.close_call_count == (
                2 if cancellation_stage == "admission_cleanup" else 1
            )
            assert not adapter.is_resource_open
            assert runtime.state == "CLOSED"
        finally:
            update_task.cancel()
            pending_tasks = [update_task]
            if queued_update_task is not None:
                queued_update_task.cancel()
                pending_tasks.append(queued_update_task)
            else:
                pass
            if client_close_task is not None:
                pending_tasks.append(client_close_task)
            else:
                pass
            await asyncio.gather(*pending_tasks, return_exceptions=True)


@pytest.mark.parametrize("cleanup_failure", ["exception", "timeout"])
@pytest.mark.asyncio
async def test_failed_admission_cleanup_rejects_queued_update_and_releases_resources(
    cleanup_failure: Literal["exception", "timeout"],
) -> None:
    """Failed admission cleanup must release resources and reject queued session updates."""

    class FailedCleanupAdapter(GatedAdapter):
        def __init__(self) -> None:
            super().__init__([])
            self.cleanup_started: asyncio.Event = asyncio.Event()
            self.is_resource_open: bool = False

        async def open(
            self, session_id: str, config: SessionConfiguration, emit: OutputSink
        ) -> None:
            self.is_resource_open = True
            raise RuntimeError("pipeline unavailable")

        async def close(self) -> None:
            await super().close()
            if self.close_call_count == 1:
                self.cleanup_started.set()
                await self.release.wait()
                raise RuntimeError("pipeline cleanup failed")
            else:
                self.is_resource_open = False

    adapter = FailedCleanupAdapter()
    factory_call_count = 0

    def create_adapter() -> InteractionAdapter:
        nonlocal factory_call_count
        factory_call_count += 1
        return adapter

    runtime = SessionRuntime(
        MODEL_NAME,
        Capabilities(),
        create_adapter,
        RuntimeLimits(cleanup_timeout_s=ADMISSION_CLEANUP_TIMEOUT_SECONDS),
    )
    async with running_runtime(runtime):
        update_task = asyncio.create_task(runtime.update({}, "open"))
        queued_update_task: asyncio.Task[None] | None = None
        try:
            await asyncio.wait_for(
                adapter.cleanup_started.wait(), TEST_WAIT_TIMEOUT_SECONDS
            )
            queued_update_task = asyncio.create_task(runtime.update({}, "queued"))
            await asyncio.sleep(0)
            if cleanup_failure == "exception":
                adapter.release.set()
            else:
                pass

            with pytest.raises(RuntimeError, match="admission cleanup failed"):
                await asyncio.wait_for(update_task, TEST_WAIT_TIMEOUT_SECONDS)
            with pytest.raises(ProtocolError, match="closing"):
                await asyncio.wait_for(queued_update_task, TEST_WAIT_TIMEOUT_SECONDS)

            envelopes = await asyncio.wait_for(
                receive_until(runtime, Closed), TEST_WAIT_TIMEOUT_SECONDS
            )
            assert closing_failure(envelopes) == ("internal", "internal")
            assert [
                envelope.event.client_event_id
                for envelope in envelopes
                if isinstance(envelope.event, Failure)
            ] == ["open"]
            assert factory_call_count == 1
            assert adapter.close_call_count == 2
            assert not adapter.is_resource_open
            assert runtime.state == "CLOSED"
        finally:
            update_task.cancel()
            pending_tasks = [update_task]
            if queued_update_task is not None:
                queued_update_task.cancel()
                pending_tasks.append(queued_update_task)
            else:
                pass
            await asyncio.gather(*pending_tasks, return_exceptions=True)


def closing_failure(envelopes: list[Envelope]) -> tuple[str, str]:
    failures = [
        envelope.event for envelope in envelopes if isinstance(envelope.event, Failure)
    ]
    closed = envelopes[-1].event
    assert len(failures) == 1 and failures[0].is_fatal and isinstance(closed, Closed)
    return failures[0].code, closed.reason
