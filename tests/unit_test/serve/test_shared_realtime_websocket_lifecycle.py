# SPDX-License-Identifier: Apache-2.0
import asyncio

import httpx
import pytest
import websockets

from sglang_omni.client import Client
from sglang_omni.proto import OmniRequest
from sglang_omni.serve.realtime.adapters import CoordinatorAdapter
from sglang_omni.serve.realtime.control import Closed, Failure
from sglang_omni.serve.realtime.manager import RealtimeDeployment
from sglang_omni.serve.realtime.output import TurnFailure
from sglang_omni.serve.realtime.runtime import (
    Capabilities,
    RuntimeLimits,
    SessionRuntime,
)
from tests.unit_test.fixtures.realtime_websocket import (
    Producer,
    append,
    collect_events,
    endpoint,
    native_sessions,
    observe_connection_release,
    observe_native_command,
    recv,
    send,
    until,
)
from tests.unit_test.fixtures.session_pipeline import pipeline


@pytest.mark.asyncio
async def test_failed_connection_releases_admission_capacity(monkeypatch):
    class Unknown(Producer):
        async def cancel(self):
            raise RuntimeError("interrupted native media consumption is unknown")

    producer = Unknown()
    producer.release.clear()
    deployment = RealtimeDeployment(Capabilities(), lambda: producer, max_connections=1)
    async with endpoint(deployment=deployment) as (_, url, _, app):
        released = observe_connection_release(monkeypatch, app.state.realtime_manager)
        async with websockets.connect(url) as ws:
            await recv(ws)
            await send(ws, "session.update", session={})
            await recv(ws)
            await append(ws, 0, 320)
            await recv(ws)
            await producer.started.wait()
            await send(ws, "response.cancel", "cancel")
            error = await recv(ws)
            assert (
                error["sglang"]["fatal"] is True
                and error["error"]["code"] == "internal"
            )
            assert (await recv(ws))["type"] == "session.closed"
        assert producer.closed == 1
        await asyncio.wait_for(released.wait(), 5)
        async with websockets.connect(url) as replacement:
            assert (await recv(replacement))["type"] == "session.created"
            await send(replacement, "session.update", session={})
            assert (await recv(replacement))["type"] == "session.updated"


@pytest.mark.asyncio
async def test_typed_context_limit_closes_with_explicit_reason():
    class Limited(Producer):
        async def process(self, unit, epoch):
            await self.emit(
                TurnFailure(
                    "server_error", "context_limit", "native context exhausted"
                ),
                epoch,
            )
            return unit.real_samples

    async with endpoint(Limited()) as (_, url, _, _):
        async with websockets.connect(url) as ws:
            await recv(ws)
            await send(ws, "session.update", session={})
            await recv(ws)
            await append(ws, 0, 320)
            _, events = await until(ws, "session.closed")
            assert events[-1]["reason"] == "context_limit"
            assert any(
                e.get("error", {}).get("code") == "context_limit" for e in events
            )


@pytest.mark.asyncio
async def test_capabilities_unready_is_503_without_allocation():
    async with endpoint() as (http, _, producer, app):
        app.state.client.health = lambda: {"running": False}
        async with httpx.AsyncClient() as client:
            response = await client.get(http + "/v1/realtime/capabilities")
        assert response.status_code == 503
        assert producer.opened == 0


@pytest.mark.asyncio
async def test_created_connections_exhaust_capacity_before_model_allocation():
    producer = Producer()
    deployment = RealtimeDeployment(Capabilities(), lambda: producer, max_connections=1)
    async with endpoint(deployment=deployment) as (_, url, _, app):
        async with websockets.connect(url) as first:
            await recv(first)
            with pytest.raises(websockets.exceptions.InvalidStatus) as error:
                async with websockets.connect(url):
                    pass
            assert error.value.response.status_code == 503
            assert len(app.state.realtime_manager.active_sessions()) == 1
            assert producer.opened == 0


@pytest.mark.asyncio
async def test_coordinator_cleanup_failure_stops_reader_but_keeps_native_owner(
    tmp_path, monkeypatch
):
    async with pipeline(tmp_path) as (coordinator, _, _):
        adapter = CoordinatorAdapter(
            Client(coordinator),
            stages=["source", "sink"],
            request_builder=lambda cfg: OmniRequest(inputs=None, params={"delay": 0.2}),
            output_converter=lambda output: [],
            atomic_consumption=True,
        )
        runtime = SessionRuntime(
            "mock",
            Capabilities(),
            lambda: adapter,
            RuntimeLimits(),
        )
        await runtime.update({}, "open")
        runtime.limits = RuntimeLimits(cleanup_timeout_s=0.2)
        adapter.set_limits(runtime.limits)
        entered = asyncio.Event()
        append_session = coordinator.append_session

        async def appended(*args, **kwargs):
            entered.set()
            return await append_session(*args, **kwargs)

        monkeypatch.setattr(coordinator, "append_session", appended)
        await runtime.append(b"\0\0" * 320, 0, 0, "append")
        await asyncio.wait_for(entered.wait(), 5)
        assert adapter.active is not None

        def fail_close(op):
            if op == "close":
                raise TimeoutError("native release unacknowledged")

        observe_native_command(monkeypatch, coordinator, fail_close)
        await runtime.close("client_closed")
        assert runtime.worker.done() and adapter.reader.done()
        retained = native_sessions(coordinator)[runtime.session_id]
        assert retained.cleanup_error is not None
        events = await collect_events(runtime)
        assert any(
            isinstance(e, Failure) and e.code == "cleanup_timeout" for e in events
        )
        assert not any(isinstance(e, Closed) for e in events)
        monkeypatch.undo()
