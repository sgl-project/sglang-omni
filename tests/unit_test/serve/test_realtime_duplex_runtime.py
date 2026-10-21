# SPDX-License-Identifier: Apache-2.0
"""Session runtime contracts that depend on scheduling the wire cannot pin."""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager
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
    OutputEvent,
    ResponseFinished,
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
from tests.unit_test.fixtures.realtime_fakes import FakeClock

MODEL_NAME = "duplex-test"
SAMPLE_RATE = 16000
NATIVE_UNIT_MS = 20
UNIT_BYTES = SAMPLE_RATE * NATIVE_UNIT_MS // 1000 * 2
RUNTIME_LOGGER_NAME = "sglang_omni.serve.realtime.runtime"
ACTIVITY_TIMEOUT_SECONDS = 0.5
DEADLINE_EPSILON_SECONDS = ACTIVITY_TIMEOUT_SECONDS / 10
TEST_WAIT_TIMEOUT_SECONDS = 5.0
ACTIVITY_LIMITS = RuntimeLimits(
    session_update_timeout_s=ACTIVITY_TIMEOUT_SECONDS,
    idle_timeout_s=ACTIVITY_TIMEOUT_SECONDS,
)


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Drive the realtime runtime clock and asyncio.sleep from one fake clock."""
    clock = FakeClock()
    monkeypatch.setattr(runtime_module, "time", clock)
    monkeypatch.setattr(asyncio, "sleep", clock.sleep)
    return clock


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


class GatedAdmissionAdapter(GatedAdapter):
    """Reports when admission starts and holds it until released."""

    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        self.has_started.set()
        await self.release.wait()
        await super().open(session_id, config, emit)


class GatedClearAdapter(GatedAdapter):
    async def clear(self) -> int:
        self.has_started.set()
        await self.release.wait()
        return 0


class CommandAdapter(GatedAdapter):
    """Supports turn commands and can pause commit until released."""

    async def commit(self) -> None:
        self.has_started.set()
        await self.release.wait()

    async def cancel(self) -> None:
        pass

    async def truncate(self, audio_end_ms: float) -> None:
        pass


class RejectedOpenAdapter(GatedAdmissionAdapter):
    def __init__(self) -> None:
        super().__init__([])
        self.cleanup_started: asyncio.Event = asyncio.Event()
        self.cleanup_release: asyncio.Event = asyncio.Event()

    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        await super().open(session_id, config, emit)
        raise RuntimeError("pipeline unavailable")

    async def close(self) -> None:
        self.cleanup_started.set()
        await self.cleanup_release.wait()


@asynccontextmanager
async def running_runtime(runtime: SessionRuntime) -> AsyncIterator[SessionRuntime]:
    runtime.start()
    try:
        yield runtime
    finally:
        await asyncio.wait_for(
            runtime.close("client_closed"), TEST_WAIT_TIMEOUT_SECONDS
        )


@asynccontextmanager
async def open_runtime(
    adapter: GatedAdapter,
    limits: RuntimeLimits | None = None,
    capabilities: Capabilities | None = None,
) -> AsyncIterator[SessionRuntime]:
    runtime = SessionRuntime(
        MODEL_NAME,
        capabilities or Capabilities(),
        lambda: adapter,
        limits or RuntimeLimits(),
    )
    async with running_runtime(runtime):
        await runtime.update({}, "client_update")
        yield runtime


async def receive_until(
    runtime: SessionRuntime, event_type: type[Drained | UnitCompleted | Closed]
) -> list[Envelope]:
    envelopes: list[Envelope] = []
    loop = asyncio.get_running_loop()
    deadline_seconds = loop.time() + TEST_WAIT_TIMEOUT_SECONDS
    async with aclosing(runtime.outputs()) as outputs:
        while True:
            envelope = await asyncio.wait_for(
                anext(outputs), deadline_seconds - loop.time()
            )
            envelopes.append(envelope)
            if isinstance(envelope.event, event_type):
                return envelopes
            else:
                pass


def ready_observation(
    timestamp_ns: int, run_id: str = "run-a"
) -> SessionUnitReadyObservation:
    return SessionUnitReadyObservation(timestamp_ns=timestamp_ns, run_id=run_id)


@pytest.mark.asyncio
async def test_eos_marks_only_the_last_unit_when_end_arrives_mid_backlog() -> None:
    adapter = GatedAdapter([])
    async with open_runtime(adapter) as runtime:
        await runtime.append(b"\1" * UNIT_BYTES, 0, None, "client_append_0")
        await asyncio.wait_for(adapter.has_started.wait(), TEST_WAIT_TIMEOUT_SECONDS)
        await runtime.append(b"\1" * UNIT_BYTES * 2, 1, None, "client_append_1")
        await runtime.end("client_end")

        adapter.release.set()
        drained = (await receive_until(runtime, Drained))[-1].event

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
    async with open_runtime(adapter) as runtime:
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
    async with open_runtime(adapter) as runtime:
        await runtime.append(b"\1" * UNIT_BYTES * 2, 0, None, "client_append_0")
        await adapter.has_started.wait()
        await runtime.end("client_end")

        adapter.release.set()
        await receive_until(runtime, Drained)
        await runtime.close("client_closed")

        assert [unit.ready_observation.timestamp_ns for unit in adapter.units] == [
            100,
            100,
        ]


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
    async with open_runtime(adapter) as runtime:
        append_bytes = UNIT_BYTES if has_empty_tail else UNIT_BYTES + UNIT_BYTES // 2
        await runtime.append(b"\1" * append_bytes, 0, None, "client_append_0")
        await adapter.has_started.wait()
        await runtime.end("client_end")

        adapter.release.set()
        await receive_until(runtime, Drained)
        await runtime.close("client_closed")

        assert [unit.ready_observation.timestamp_ns for unit in adapter.units] == [
            100,
            200,
        ]
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
    async with open_runtime(adapter) as runtime:
        await runtime.append(b"\1" * UNIT_BYTES, 0, None, "client_append_0")
        await asyncio.wait_for(adapter.has_started.wait(), TEST_WAIT_TIMEOUT_SECONDS)
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
        async with open_runtime(FailingAdapter([])) as runtime:
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
    fake_clock: FakeClock,
) -> None:
    adapter = GatedAdmissionAdapter([])
    runtime = SessionRuntime(
        MODEL_NAME,
        Capabilities(),
        lambda: adapter,
        ACTIVITY_LIMITS,
    )
    async with running_runtime(runtime):
        update_task = asyncio.create_task(runtime.update({}, "update"))
        await adapter.has_started.wait()
        await fake_clock.advance(ACTIVITY_TIMEOUT_SECONDS * 2)
        assert not update_task.done()
        assert runtime.state == "CREATED"
        adapter.release.set()
        await update_task
        assert runtime.state == "OPEN"
        await runtime.close("client_closed")
        envelopes = await receive_until(runtime, Closed)

    assert [
        entry.event for entry in envelopes if isinstance(entry.event, Failure)
    ] == []
    assert envelopes[-1].event == Closed("client_closed")


@pytest.mark.asyncio
async def test_update_after_the_deadline_is_rejected_without_admission(
    fake_clock: FakeClock,
) -> None:
    adapter = GatedAdmissionAdapter([])
    adapter.release.set()
    runtime = SessionRuntime(
        MODEL_NAME,
        Capabilities(),
        lambda: adapter,
        ACTIVITY_LIMITS,
    )
    async with running_runtime(runtime):
        await fake_clock.advance(0)
        # note (Codex): Queue the command before waking the watchdog to exercise the admission fence.
        update_task = asyncio.create_task(runtime.update({}, "update"))
        await fake_clock.advance(ACTIVITY_TIMEOUT_SECONDS + DEADLINE_EPSILON_SECONDS)
        await update_task
        envelopes = await receive_until(runtime, Closed)

    assert not adapter.has_started.is_set()
    failures = [entry.event for entry in envelopes if isinstance(entry.event, Failure)]
    assert [failure.code for failure in failures] == ["session_update_timeout"]
    assert closing_failure(envelopes) == (
        "session_update_timeout",
        "session_update_timeout",
    )
    assert runtime.state == "CLOSED"


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
    async with open_runtime(GatedAdapter([])) as runtime:
        runtime.fail("context full", "context_exhausted")
        await asyncio.wait_for(
            asyncio.gather(runtime.close("disconnect"), runtime.close("client_closed")),
            TEST_WAIT_TIMEOUT_SECONDS,
        )

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
    async with open_runtime(adapter) as runtime:
        await runtime.append(b"\1" * UNIT_BYTES, 0, None, "append")
        await asyncio.wait_for(adapter.has_started.wait(), TEST_WAIT_TIMEOUT_SECONDS)
        if termination == "cancelled_task":
            assert runtime.pump_task is not None
            runtime.pump_task.cancel()
        elif termination == "client_close":
            await runtime.close("client_closed")
        else:
            pass

        envelopes = await receive_until(runtime, Closed)
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

    adapter = GatedAdmissionAdapter([])
    runtime = SessionRuntime(
        MODEL_NAME, Capabilities(), lambda: adapter, RuntimeLimits()
    )
    runtime.start()
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

            envelopes = await receive_until(runtime, Closed)
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
        RuntimeLimits(cleanup_timeout_s=ACTIVITY_TIMEOUT_SECONDS),
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

            envelopes = await receive_until(runtime, Closed)
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


async def assert_idle_closed(runtime: SessionRuntime) -> None:
    envelopes = await receive_until(runtime, Closed)
    assert closing_failure(envelopes) == ("idle_timeout", "idle_timeout")
    assert runtime.state == "CLOSED"


async def advance_past_deadline(
    clock: FakeClock, runtime: SessionRuntime, deadline_seconds: float
) -> None:
    """Check the session is still open just before the deadline, then move past it."""
    await clock.advance(deadline_seconds - DEADLINE_EPSILON_SECONDS)
    assert runtime.state == "OPEN"
    await clock.advance(2 * DEADLINE_EPSILON_SECONDS)


@pytest.mark.asyncio
async def test_slow_pipeline_open_is_not_an_admission_timeout(
    fake_clock: FakeClock,
) -> None:
    adapter = GatedAdmissionAdapter([])
    runtime = SessionRuntime(
        MODEL_NAME, Capabilities(), lambda: adapter, ACTIVITY_LIMITS
    )
    async with running_runtime(runtime):
        update_task = asyncio.create_task(runtime.update({}, "client_update"))
        await adapter.has_started.wait()
        await fake_clock.advance(ACTIVITY_TIMEOUT_SECONDS * 2)
        assert not update_task.done()
        adapter.release.set()
        await update_task
        assert runtime.state == "OPEN"
        await advance_past_deadline(fake_clock, runtime, ACTIVITY_TIMEOUT_SECONDS)
        await assert_idle_closed(runtime)


@pytest.mark.parametrize("cleanup_delay_seconds", [0, ACTIVITY_TIMEOUT_SECONDS * 1.25])
@pytest.mark.asyncio
async def test_failed_admission_after_deadline_closes_without_retry(
    fake_clock: FakeClock,
    cleanup_delay_seconds: float,
) -> None:
    """An admission that fails after the deadline must close the session instead of accepting a retry."""
    adapter = RejectedOpenAdapter()
    if cleanup_delay_seconds == 0:
        adapter.cleanup_release.set()
    else:
        pass
    runtime = SessionRuntime(
        MODEL_NAME, Capabilities(), lambda: adapter, ACTIVITY_LIMITS
    )
    async with running_runtime(runtime):
        update_task = asyncio.create_task(runtime.update({}, "client_update"))
        await adapter.has_started.wait()
        await fake_clock.advance(ACTIVITY_TIMEOUT_SECONDS * 2.5)
        adapter.release.set()
        await adapter.cleanup_started.wait()
        await fake_clock.advance(cleanup_delay_seconds)
        adapter.cleanup_release.set()
        await update_task
        with pytest.raises(ProtocolError, match="closing"):
            await runtime.update({}, "client_retry")
        envelopes = await receive_until(runtime, Closed)
        assert closing_failure(envelopes) == (
            "session_update_timeout",
            "session_update_timeout",
        )


@pytest.mark.asyncio
async def test_repeated_failed_admissions_keep_original_deadline(
    fake_clock: FakeClock,
) -> None:
    """Failed admissions must not restart the deadline, so retries cannot keep an unadmitted session open."""

    class RejectedAdapter(GatedAdapter):
        async def open(
            self, session_id: str, config: SessionConfiguration, emit: OutputSink
        ) -> None:
            raise RuntimeError("pipeline unavailable")

    factory_call_count = 0

    def create_adapter() -> InteractionAdapter:
        nonlocal factory_call_count
        factory_call_count += 1
        return RejectedAdapter([])

    runtime = SessionRuntime(
        MODEL_NAME, Capabilities(), create_adapter, ACTIVITY_LIMITS
    )
    async with running_runtime(runtime):
        for attempt_index in range(3):
            await fake_clock.advance(
                ACTIVITY_TIMEOUT_SECONDS * 0.4 if attempt_index else 0
            )
            with pytest.raises(ProtocolError) as rejection:
                await runtime.update({}, f"attempt_{attempt_index}")
            assert rejection.value.code == "admission_rejected"
        update_task = asyncio.create_task(runtime.update({}, "late_retry"))
        await fake_clock.advance(ACTIVITY_TIMEOUT_SECONDS * 0.4)
        await update_task
        with pytest.raises(ProtocolError, match="closing"):
            await runtime.update({}, "after_close")

        envelopes = await receive_until(runtime, Closed)
        assert closing_failure(envelopes) == (
            "session_update_timeout",
            "session_update_timeout",
        )
        assert factory_call_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("is_valid_image", [False, True])
async def test_only_accepted_images_extend_idle_timeout(
    fake_clock: FakeClock,
    is_valid_image: bool,
) -> None:
    adapter = GatedAdapter([])
    async with open_runtime(
        adapter, ACTIVITY_LIMITS, Capabilities(input_modalities=("audio", "image"))
    ) as runtime:
        await fake_clock.advance(ACTIVITY_TIMEOUT_SECONDS / 2)
        if is_valid_image:
            await runtime.append_image(b"\xff\xd8frame", 0, "client_image")
        else:
            with pytest.raises(ProtocolError, match="image must be JPEG or PNG"):
                await runtime.append_image(b"invalid", 0, "client_image")

        await advance_past_deadline(fake_clock, runtime, ACTIVITY_TIMEOUT_SECONDS / 2)
        if is_valid_image:
            await advance_past_deadline(
                fake_clock,
                runtime,
                ACTIVITY_TIMEOUT_SECONDS / 2 - DEADLINE_EPSILON_SECONDS,
            )
        else:
            pass
        await assert_idle_closed(runtime)


@pytest.mark.asyncio
async def test_steady_input_keeps_session_open_past_idle_timeout(
    fake_clock: FakeClock,
) -> None:
    adapter = GatedAdapter([])
    async with open_runtime(adapter, ACTIVITY_LIMITS) as runtime:
        # Note (Haiyang Luo): One sample updates input activity without a complete input unit.
        for sequence in range(3):
            await runtime.append(b"\1\0", sequence, None, f"client_append_{sequence}")
            await fake_clock.advance(
                ACTIVITY_TIMEOUT_SECONDS - DEADLINE_EPSILON_SECONDS
            )
            assert runtime.state == "OPEN"

        assert runtime.state == "OPEN" and not adapter.units
        await fake_clock.advance(2 * DEADLINE_EPSILON_SECONDS)
        await assert_idle_closed(runtime)


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["commit", "cancel", "truncate"])
async def test_client_commands_restart_idle_timeout(
    fake_clock: FakeClock,
    command: Literal["commit", "cancel", "truncate"],
) -> None:
    """Successful client turn commands keep an otherwise idle session open."""
    adapter = CommandAdapter([])
    adapter.release.set()
    async with open_runtime(
        adapter, ACTIVITY_LIMITS, Capabilities(supports_client_commit=True)
    ) as runtime:
        await fake_clock.advance(ACTIVITY_TIMEOUT_SECONDS * 0.75)
        if command == "commit":
            await runtime.commit()
        elif command == "cancel":
            await runtime.cancel()
        else:
            await runtime.truncate(0.0)
        await fake_clock.advance(0)

        await fake_clock.advance(ACTIVITY_TIMEOUT_SECONDS * 0.75)
        assert runtime.state == "OPEN"
        await fake_clock.advance(ACTIVITY_TIMEOUT_SECONDS * 0.5)
        await assert_idle_closed(runtime)


@pytest.mark.parametrize("command", ["commit", "end"])
@pytest.mark.asyncio
async def test_command_accepted_before_idle_deadline_reaches_adapter_when_watchdog_runs_first(
    fake_clock: FakeClock,
    command: Literal["commit", "end"],
) -> None:
    """An accepted commit or end of input must survive the old idle deadline while awaiting the pump."""
    adapter = CommandAdapter([])
    async with open_runtime(
        adapter, ACTIVITY_LIMITS, Capabilities(supports_client_commit=True)
    ) as runtime:
        await fake_clock.advance(ACTIVITY_TIMEOUT_SECONDS - DEADLINE_EPSILON_SECONDS)
        [(watchdog_deadline_seconds, watchdog_wakeup)] = fake_clock.sleepers
        assert watchdog_deadline_seconds == ACTIVITY_TIMEOUT_SECONDS
        # note (Codex): Queue the watchdog first to model a delay across the deadline.
        watchdog_wakeup.set_result(None)
        if command == "commit":
            await runtime.commit()
        else:
            await runtime.end("end")
        assert not adapter.has_started.is_set()

        await fake_clock.advance(2 * DEADLINE_EPSILON_SECONDS)

        assert runtime.state == "OPEN"
        assert adapter.has_started.is_set()
        adapter.release.set()
        await fake_clock.advance(0)


@pytest.mark.asyncio
async def test_asynchronous_output_restarts_idle_timeout(fake_clock: FakeClock) -> None:
    """Replies streaming after input stops keep the session open until output stops."""
    adapter = GatedAdapter([])
    async with open_runtime(adapter, ACTIVITY_LIMITS) as runtime:
        assert adapter.output_sink is not None
        await fake_clock.advance(ACTIVITY_TIMEOUT_SECONDS * 0.75)
        await adapter.output_sink(ResponseStarted("response_0"))
        await fake_clock.advance(ACTIVITY_TIMEOUT_SECONDS * 0.75)
        assert runtime.state == "OPEN"

        await adapter.output_sink(TextDelta("response_0", "item_0", "reply"))
        await fake_clock.advance(ACTIVITY_TIMEOUT_SECONDS * 0.75)
        assert runtime.state == "OPEN"
        await fake_clock.advance(ACTIVITY_TIMEOUT_SECONDS * 0.5)
        await assert_idle_closed(runtime)


@pytest.mark.asyncio
async def test_commit_in_progress_is_protected_and_completion_restarts_idle_window(
    fake_clock: FakeClock,
) -> None:
    """A commit without audio stays active while blocked and gets a fresh idle window."""
    adapter = CommandAdapter([])
    async with open_runtime(
        adapter, ACTIVITY_LIMITS, Capabilities(supports_client_commit=True)
    ) as runtime:
        await fake_clock.advance(ACTIVITY_TIMEOUT_SECONDS * 0.75)
        await runtime.commit()
        await adapter.has_started.wait()
        await fake_clock.advance(ACTIVITY_TIMEOUT_SECONDS * 1.5)
        assert runtime.state == "OPEN"

        await fake_clock.advance(ACTIVITY_TIMEOUT_SECONDS * 0.5)
        adapter.release.set()
        await fake_clock.advance(0)
        await fake_clock.advance(
            ACTIVITY_TIMEOUT_SECONDS * 0.75 + DEADLINE_EPSILON_SECONDS
        )
        assert runtime.state == "OPEN"
        await fake_clock.advance(ACTIVITY_TIMEOUT_SECONDS * 0.25)
        await assert_idle_closed(runtime)


@pytest.mark.parametrize(
    "idle_timeout_seconds", [ACTIVITY_TIMEOUT_SECONDS / 2, ACTIVITY_TIMEOUT_SECONDS * 2]
)
@pytest.mark.asyncio
async def test_clear_in_progress_is_protected_and_completion_restarts_idle_window(
    fake_clock: FakeClock,
    idle_timeout_seconds: float,
) -> None:
    adapter = GatedClearAdapter([])
    async with open_runtime(
        adapter,
        RuntimeLimits(idle_timeout_s=idle_timeout_seconds),
    ) as runtime:
        await fake_clock.advance(idle_timeout_seconds * 0.75)
        clear_task = asyncio.create_task(runtime.clear("client_clear"))
        await adapter.has_started.wait()
        # note (Haiyang Luo): The watchdog wakes at the old deadline, sees the held lock, and sleeps one more period.
        await fake_clock.advance(idle_timeout_seconds * 0.5)
        assert not clear_task.done()
        assert runtime.state == "OPEN"

        await fake_clock.advance(idle_timeout_seconds * 0.25)
        adapter.release.set()
        await clear_task
        # note (Haiyang Luo): The watchdog wakes again here; only the clear completion keeps the session open.
        await fake_clock.advance(idle_timeout_seconds * 0.75 + DEADLINE_EPSILON_SECONDS)
        assert runtime.state == "OPEN"
        await fake_clock.advance(idle_timeout_seconds * 0.25)
        await assert_idle_closed(runtime)


@pytest.mark.asyncio
async def test_unit_in_flight_is_not_idle_and_idle_resumes_after_it(
    fake_clock: FakeClock,
) -> None:
    adapter = GatedAdapter([])
    async with open_runtime(adapter, ACTIVITY_LIMITS) as runtime:
        await runtime.append(b"\1" * UNIT_BYTES, 0, None, "client_append_0")
        await asyncio.wait_for(adapter.has_started.wait(), TEST_WAIT_TIMEOUT_SECONDS)

        await fake_clock.advance(ACTIVITY_TIMEOUT_SECONDS * 3)
        assert runtime.state == "OPEN"

        adapter.release.set()
        completed = (await receive_until(runtime, UnitCompleted))[-1].event
        assert completed == UnitCompleted(adapter.units[0].unit_id)
        await advance_past_deadline(fake_clock, runtime, ACTIVITY_TIMEOUT_SECONDS)
        await assert_idle_closed(runtime)


@pytest.mark.asyncio
async def test_unit_completing_as_idle_deadline_expires_keeps_session_open(
    fake_clock: FakeClock,
) -> None:
    adapter = GatedAdapter([])
    async with open_runtime(adapter, ACTIVITY_LIMITS) as runtime:
        await runtime.append(b"\1" * UNIT_BYTES, 0, None, "client_append_0")
        await asyncio.wait_for(adapter.has_started.wait(), TEST_WAIT_TIMEOUT_SECONDS)
        await fake_clock.advance(0)

        # note (Codex): Releasing first queues completion ahead of the expired watchdog.
        adapter.release.set()
        await fake_clock.advance(ACTIVITY_TIMEOUT_SECONDS * 1.5)
        completed = (await receive_until(runtime, UnitCompleted))[-1].event
        assert completed == UnitCompleted(adapter.units[0].unit_id)

        await fake_clock.advance(ACTIVITY_TIMEOUT_SECONDS / 4)
        assert runtime.state == "OPEN"


@pytest.mark.asyncio
async def test_idle_timeout_shorter_than_admission_is_honored_after_open(
    fake_clock: FakeClock,
) -> None:
    limits = RuntimeLimits(
        session_update_timeout_s=ACTIVITY_TIMEOUT_SECONDS * 4,
        idle_timeout_s=ACTIVITY_TIMEOUT_SECONDS,
    )
    runtime = SessionRuntime(
        MODEL_NAME, Capabilities(), lambda: GatedAdapter([]), limits
    )
    async with running_runtime(runtime):
        await fake_clock.advance(ACTIVITY_TIMEOUT_SECONDS / 8)
        await runtime.update({}, "client_update")
        await advance_past_deadline(fake_clock, runtime, ACTIVITY_TIMEOUT_SECONDS)
        await assert_idle_closed(runtime)
