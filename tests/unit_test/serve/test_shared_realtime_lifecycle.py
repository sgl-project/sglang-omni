# SPDX-License-Identifier: Apache-2.0
import asyncio
import json

import pytest

from sglang_omni.proto.session import SessionRef
from sglang_omni.serve.realtime.adapters import CoordinatorAdapter
from sglang_omni.serve.realtime.control import Closed, Failure, Updated
from sglang_omni.serve.realtime.output import ResponseStarted, TextDelta
from sglang_omni.serve.realtime.protocol import SharedRealtimeSession
from sglang_omni.serve.realtime.runtime import (
    Capabilities,
    RuntimeLimits,
    SessionRuntime,
    Unit,
)
from tests.unit_test.fixtures.realtime_websocket import (
    Adapter,
    collect_events,
    observe_close_entry,
)


@pytest.mark.asyncio
async def test_output_overflow_failure_has_bounded_deliverable_terminal(monkeypatch):
    class ScenarioAdapter(Adapter):
        async def process(self, unit, epoch):
            await self.emit(ResponseStarted("r"), epoch)
            for _ in range(10):
                await self.emit(TextDelta("r", "i", "chunk"), epoch)
            return unit.real_samples

    runtime = SessionRuntime(
        "mock", Capabilities(), ScenarioAdapter, RuntimeLimits(max_output_events=3)
    )
    failed = asyncio.Event()
    original = runtime.fail

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        failed.set()

    monkeypatch.setattr(runtime, "fail", fail)
    await runtime.update({}, "open")
    await runtime.append(b"\0\0" * 320, 0, 0, "append")
    await asyncio.wait_for(failed.wait(), 5)
    await runtime.close("client_closed")
    events = await collect_events(runtime)
    assert len(events) == 2
    assert isinstance(events[0], Failure) and isinstance(events[1], Closed)


@pytest.mark.asyncio
async def test_explicit_close_drains_stalled_sender_before_socket_close(monkeypatch):

    class Socket:
        def __init__(self):
            self.blocked = asyncio.Event()
            self.release = asyncio.Event()
            self.frames = []

        async def receive(self):
            await self.blocked.wait()
            return {
                "type": "websocket.receive",
                "text": json.dumps({"type": "session.close", "event_id": "close"}),
            }

        async def send_text(self, raw):
            event = json.loads(raw)
            if event["type"] == "response.created":
                self.blocked.set()
                await self.release.wait()
            self.frames.append(event)

        async def close(self):
            self.frames.append({"type": "socket.closed"})

    runtime = SessionRuntime("mock", Capabilities(), Adapter, RuntimeLimits())
    await runtime.update({}, "open")
    await runtime.emit(ResponseStarted("r"), 0)
    socket = Socket()
    session = SharedRealtimeSession(socket, runtime)
    close_entered = observe_close_entry(monkeypatch, runtime)
    task = asyncio.create_task(session.run())
    await socket.blocked.wait()
    await asyncio.wait_for(close_entered.wait(), 5)
    assert not task.done()
    socket.release.set()
    await asyncio.wait_for(task, 1)
    kinds = [event["type"] for event in socket.frames]
    assert (
        kinds.index("response.done")
        < kinds.index("session.closed")
        < kinds.index("socket.closed")
    )


@pytest.mark.asyncio
async def test_close_during_delayed_admission_cannot_resurrect_or_leak(monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()

    class ScenarioAdapter(Adapter):
        allocated = False

        async def open(self, *args):
            started.set()
            await release.wait()
            self.allocated = True

        async def close(self):
            self.allocated = False

    adapter = ScenarioAdapter()
    runtime = SessionRuntime("mock", Capabilities(), lambda: adapter, RuntimeLimits())
    notifications = []
    notify = runtime.notify

    def notified(event):
        notifications.append(event)
        return notify(event)

    monkeypatch.setattr(runtime, "notify", notified)
    opening = asyncio.create_task(runtime.update({}, "open"))
    await started.wait()
    close_entered = observe_close_entry(monkeypatch, runtime)
    closing = asyncio.create_task(runtime.close("session_timeout"))
    try:
        await asyncio.wait_for(close_entered.wait(), 5)
    finally:
        release.set()
    await asyncio.wait_for(asyncio.gather(opening, closing), 5)
    assert runtime.state == "CLOSED"
    assert not adapter.allocated
    assert not any(isinstance(event, Updated) for event in notifications)
    assert runtime.worker is None or runtime.worker.done()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "active,close_failure",
    [(False, "timeout"), (True, "timeout"), (True, "exception")],
    ids=["idle_timeout", "active_timeout", "active_exception"],
)
async def test_failed_close_reports_cleanup_timeout_and_stops_worker(
    active, close_failure
):
    entered = asyncio.Event()

    class ScenarioAdapter(Adapter):
        async def process(self, *args):
            entered.set()
            await asyncio.Event().wait()

        async def close(self):
            if close_failure == "exception":
                raise RuntimeError("release unacknowledged")
            await asyncio.Event().wait()

    runtime = SessionRuntime(
        "mock", Capabilities(), ScenarioAdapter, RuntimeLimits(cleanup_timeout_s=0.02)
    )
    await runtime.update({}, "open")
    if active:
        await runtime.append(b"\0\0" * 320, 0, 0, "append")
        await entered.wait()
    await runtime.close("client_closed")
    assert runtime.worker.done()
    events = await collect_events(runtime)
    assert any(isinstance(e, Failure) and e.code == "cleanup_timeout" for e in events)
    assert not any(isinstance(e, Closed) for e in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("during_unit", [False, True])
async def test_coordinator_output_eof_fails_pending_and_future_units(during_unit):
    release = asyncio.Event()
    appended = asyncio.Event()
    events = []

    class Client:
        async def open_session(self, *args, **kwargs):
            return SessionRef(kwargs["session_id"])

        async def append_session(self, *args):
            appended.set()

        async def session_outputs(self, ref):
            await release.wait()
            if False:
                yield None

        async def close_session(self, ref):
            release.set()

    async def emit(event, *args):
        events.append(event)

    adapter = CoordinatorAdapter(
        Client(),
        stages=["mock"],
        request_builder=lambda cfg: None,
        output_converter=lambda output: [],
        atomic_consumption=True,
    )
    await adapter.open("eof", {}, emit)
    task = None
    try:
        unit = Unit(0, 0, b"\0\0" * 8, 8)
        if during_unit:
            task = asyncio.create_task(adapter.process(unit, 0))
            await appended.wait()
        release.set()
        await adapter.reader
        if task is None:
            task = asyncio.create_task(adapter.process(unit, 0))
        with pytest.raises(RuntimeError, match="output stream closed"):
            await asyncio.wait_for(task, 0.2)
        with pytest.raises(RuntimeError, match="output stream closed"):
            await asyncio.wait_for(adapter.process(unit, 0), 0.2)
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await adapter.close()
