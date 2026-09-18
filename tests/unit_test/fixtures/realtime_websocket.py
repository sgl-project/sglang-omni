# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
import json
import socket
from contextlib import asynccontextmanager

import uvicorn

from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.realtime.manager import RealtimeDeployment
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    AudioFinished,
    ResponseFinished,
    ResponseStarted,
    TextDelta,
    TextFinished,
)
from sglang_omni.serve.realtime.runtime import (
    Capabilities,
    InteractionAdapter,
    RuntimeLimits,
)


class Adapter(InteractionAdapter):
    async def open(self, session_id, config, emit):
        self.session_id, self.emit = session_id, emit

    async def process(self, unit, epoch):
        return unit.real_samples

    async def cancel(self):
        pass

    async def close(self):
        pass


async def next_event(outputs, event_type):
    async with asyncio.timeout(5):
        async for envelope in outputs:
            if isinstance(envelope.event, event_type):
                return envelope
    raise AssertionError(f"missing {event_type.__name__}")


async def collect_events(runtime):
    async with asyncio.timeout(5):
        return [envelope.event async for envelope in runtime.outputs()]


class Producer(Adapter):
    def __init__(self):
        self.opened = self.closed = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.units = []
        self.emit = None

    async def open(self, session_id, config, emit):
        self.opened += 1
        self.session_id = session_id
        self.emit = emit

    async def process(self, unit, epoch):
        self.units.append(unit)
        self.started.set()
        await self.release.wait()
        rid, item = f"r{unit.seq}", f"i{unit.seq}"
        await self.emit(ResponseStarted(rid), epoch)
        await self.emit(TextDelta(rid, item, "hello"), epoch)
        await self.emit(AudioDelta(rid, item, b"\0\0" * 160), epoch)
        await self.emit(TextFinished(rid, item, "hello"), epoch)
        await self.emit(AudioFinished(rid, item), epoch)
        await self.emit(
            ResponseFinished(rid, item, "hello", True, "completed", "stop"), epoch
        )
        return unit.real_samples

    async def close(self):
        self.release.set()
        self.closed += 1


@asynccontextmanager
async def endpoint(
    producer=None, *, caps=None, limits=None, deployment=None, default=False
):
    producer = producer or Producer()
    deployment = deployment or RealtimeDeployment(
        caps or Capabilities(output_modalities=("text", "audio")),
        lambda: producer,
        limits or RuntimeLimits(),
    )
    from types import SimpleNamespace

    app = create_app(
        SimpleNamespace(health=lambda: {"running": True}),
        model_name="mock",
        realtime_deployment=None if default else deployment,
        enable_realtime=default,
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    ready = asyncio.Event()
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    startup = server.startup

    async def started(*args, **kwargs):
        await startup(*args, **kwargs)
        ready.set()

    server.startup = started
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        await asyncio.wait_for(ready.wait(), 5)
        yield f"http://127.0.0.1:{port}", f"ws://127.0.0.1:{port}/v1/realtime?model=mock", producer, app
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 5)
        sock.close()


async def send(ws, kind, event_id="command", **fields):
    await ws.send(json.dumps(dict(type=kind, event_id=event_id, **fields)))


async def recv(ws):
    return json.loads(await asyncio.wait_for(ws.recv(), 2))


async def until(ws, kind):
    events = []
    while True:
        event = await recv(ws)
        events.append(event)
        if event["type"] == kind:
            return event, events


async def append(ws, seq, samples=80, start=None):
    extension = dict(seq=seq)
    if start is not None:
        extension["t_start_ms"] = start
    await send(
        ws,
        "input_audio_buffer.append",
        f"a{seq}",
        audio=base64.b64encode(b"\0\0" * samples).decode(),
        sglang=extension,
    )


def observe_close_entry(monkeypatch, runtime):
    # Note (Junnan Li): Public close hides entry to the admission-lock wait.
    entered = asyncio.Event()
    original = runtime._close

    async def close(*args, **kwargs):
        entered.set()
        return await original(*args, **kwargs)

    monkeypatch.setattr(runtime, "_close", close)
    return entered


def observe_connection_release(monkeypatch, manager):
    released = asyncio.Event()
    original = manager.close

    async def close(session_id):
        await original(session_id)
        released.set()

    monkeypatch.setattr(manager, "close", close)
    return released


def native_sessions(coordinator):
    # Note (Junnan Li): No public query exposes retained native cleanup ownership.
    return coordinator._sessions


def observe_native_command(monkeypatch, coordinator, observe):
    # Note (Junnan Li): The command boundary is below public close error handling.
    original = coordinator._session_command

    async def command(session, op, **kwargs):
        observe(op)
        return await original(session, op, **kwargs)

    monkeypatch.setattr(coordinator, "_session_command", command)
