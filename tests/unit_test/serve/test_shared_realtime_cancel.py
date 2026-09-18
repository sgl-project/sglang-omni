# SPDX-License-Identifier: Apache-2.0
import asyncio
import json

import pytest

from sglang_omni.proto.session import OutputChunk, SessionRef
from sglang_omni.serve.realtime.adapters import CoordinatorAdapter
from sglang_omni.serve.realtime.control import Cancelled
from sglang_omni.serve.realtime.output import (
    ResponseFinished,
    ResponseStarted,
    TextDelta,
)
from sglang_omni.serve.realtime.protocol import SharedRealtimeSession
from sglang_omni.serve.realtime.runtime import (
    Capabilities,
    RuntimeLimits,
    SessionRuntime,
    Unit,
)
from tests.unit_test.fixtures.realtime_websocket import Adapter, next_event


@pytest.mark.asyncio
async def test_cancel_terminates_visible_response_before_ack_and_fences_old_data():
    runtime = SessionRuntime("mock", Capabilities(), Adapter, RuntimeLimits())
    await runtime.update({}, "open")
    await runtime.emit(ResponseStarted("r"), 0)
    outputs = runtime.outputs()
    created = await next_event(outputs, ResponseStarted)
    runtime.before_send(created)
    await runtime.emit(ResponseFinished("r", "i", "", False, "completed", "stop"), 0)
    await runtime.cancel("cancel")
    await runtime.emit(TextDelta("r", "i", "late"), 0)
    envelopes = []
    async with asyncio.timeout(5):
        async for envelope in outputs:
            envelopes.append(envelope)
            if isinstance(envelope.event, Cancelled):
                break
    control = [env.event for env in envelopes if env.control]
    assert isinstance(control[-2], ResponseFinished)
    assert control[-2].status == "cancelled"
    assert isinstance(control[-1], Cancelled)
    await runtime.emit(ResponseStarted("new"), 1)
    assert isinstance((await anext(outputs)).event, ResponseStarted)
    await runtime.close("client_closed")
    await outputs.aclose()


@pytest.mark.asyncio
async def test_cancel_requires_completed_unit_receipt():
    class Client:
        async def open_session(self, *args, **kwargs):
            return SessionRef(kwargs["session_id"])

        async def append_session(self, *args):
            appended.set()

        async def session_outputs(self, ref):
            await closed.wait()
            if False:
                yield None

        async def abort_session(self, ref):
            return SessionRef(ref.session_id, ref.incarnation, ref.epoch + 1)

        async def close_session(self, ref):
            closed.set()

    closed = asyncio.Event()
    appended = asyncio.Event()
    adapter = CoordinatorAdapter(
        Client(),
        stages=["mock"],
        request_builder=lambda cfg: None,
        output_converter=lambda out: [],
        atomic_consumption=True,
    )
    await adapter.open("test", {}, lambda *args: None)
    process = asyncio.create_task(adapter.process(Unit(0, 0, b"\0\0" * 8, 8), 0))
    await asyncio.wait_for(appended.wait(), 5)
    adapter.local_cleanup_timeout = 0.02
    with pytest.raises(RuntimeError, match="receipt is missing"):
        await adapter.cancel()
    process.cancel()
    await asyncio.gather(process, return_exceptions=True)
    await adapter.close()


@pytest.mark.asyncio
async def test_native_completion_before_receipt_observation_is_consumed():
    release, closed = asyncio.Event(), asyncio.Event()

    class Client:
        async def open_session(self, *args, **kwargs):
            return SessionRef(kwargs["session_id"])

        async def append_session(self, *args):
            appended.set()

        async def session_outputs(self, ref):
            await release.wait()
            yield OutputChunk(ref, 0, 0, "audio", 0, 0.5, None, kind="input_done")
            await closed.wait()

        async def abort_session(self, ref):
            release.set()  # Note (Junnan Li): Abort can complete before its old-epoch receipt reaches the reader.
            return SessionRef(ref.session_id, 1, 1)

        async def close_session(self, ref):
            closed.set()

    appended = asyncio.Event()
    adapter = CoordinatorAdapter(
        Client(),
        stages=["mock"],
        request_builder=lambda cfg: None,
        output_converter=lambda out: [],
        atomic_consumption=True,
    )
    await adapter.open("test", {}, lambda *args: None)
    task = asyncio.create_task(adapter.process(Unit(0, 0, b"\0\0" * 8, 8), 0))
    await asyncio.wait_for(appended.wait(), 5)
    await adapter.cancel()
    assert await task == 8
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_type", ["response.created", "response.done"])
async def test_cancel_during_stalled_lifecycle_send(blocked_type):

    class Socket:
        def __init__(self):
            self.blocked = asyncio.Event()
            self.release = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.frames = []

        async def send_text(self, raw):
            frame = json.loads(raw)
            if frame["type"] == blocked_type:
                self.blocked.set()
                await self.release.wait()
            self.frames.append(frame)
            if frame["type"] == "sglang.response.cancelled":
                self.cancelled.set()

        async def close(self):
            pass

    runtime = SessionRuntime("mock", Capabilities(), Adapter, RuntimeLimits())
    await runtime.update({}, "open")
    await runtime.emit(ResponseStarted("r"), 0)
    if blocked_type == "response.done":
        await runtime.emit(
            ResponseFinished("r", "i", "", False, "completed", "stop"), 0
        )
    socket = Socket()
    session = SharedRealtimeSession(socket, runtime)
    disconnected = asyncio.Event()

    async def receive():
        await disconnected.wait()
        return {"type": "websocket.disconnect"}

    socket.receive = receive
    sender = asyncio.create_task(session.run())
    await socket.blocked.wait()
    await runtime.cancel("cancel")
    socket.release.set()
    await asyncio.wait_for(socket.cancelled.wait(), 1)
    kinds = [e["type"] for e in socket.frames]
    assert kinds.count("response.done") == 1
    assert (
        kinds.index("response.created")
        < kinds.index("response.done")
        < kinds.index("sglang.response.cancelled")
    )
    await runtime.close("client_closed")
    disconnected.set()
    await sender
