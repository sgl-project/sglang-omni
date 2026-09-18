# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import socket
from types import SimpleNamespace

import httpx
import pytest
import websockets

from sglang_omni.client import Client
from sglang_omni.proto import OmniRequest
from sglang_omni.serve.realtime import smart_turn
from sglang_omni.serve.realtime.adapters import (
    CoordinatorAdapter,
    TurnBasedAdapterFactory,
)
from sglang_omni.serve.realtime.manager import RealtimeDeployment
from sglang_omni.serve.realtime.output import (
    ResponseFinished,
    ResponseStarted,
    TextDelta,
    TextFinished,
)
from sglang_omni.serve.realtime.runtime import Capabilities, RuntimeLimits
from tests.unit_test.fixtures.realtime_websocket import (
    append,
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
async def test_shared_turn_adapter_uses_existing_completion_and_retained_clear():

    class Client:
        def __init__(self):
            self.requests = []

        async def completion_stream(self, request, **kwargs):
            self.requests.append(request)
            yield SimpleNamespace(
                modality="text", text="spoken", finish_reason="stop", usage=None
            )

        async def abort(self, request_id):
            pass

    client = Client()
    limits = RuntimeLimits(max_input_bytes=800)
    deployment = RealtimeDeployment(
        Capabilities(interaction="turn_based"),
        TurnBasedAdapterFactory(client, "mock"),
        limits,
    )
    async with endpoint(deployment=deployment) as (_, url, _, app):
        async with websockets.connect(url) as ws:
            await recv(ws)
            await send(
                ws,
                "session.update",
                session=dict(audio=dict(input=dict(turn_detection=None))),
            )
            await recv(ws)
            await append(ws, 0, 320)
            await recv(ws)
            await until(ws, "sglang.unit.done")
            assert not client.requests
            await append(ws, 1, 160)
            error, _ = await until(ws, "error")
            assert error["error"]["code"] == "buffer_overflow"
            await send(ws, "input_audio_buffer.clear")
            assert (await recv(ws))["sglang"]["discarded_ms"] == 20
            await append(ws, 1, 80, start=20)
            await recv(ws)
            await send(ws, "sglang.input_audio.end")
            drained, events = await until(ws, "sglang.input_audio.drained")
            assert drained["consumed_ms"] == 5 and drained["discarded_ms"] == 20
            assert len(client.requests) == 2
            assert any(e["type"] == "response.output_text.delta" for e in events)


@pytest.mark.asyncio
async def test_coordinator_native_public_unit_commit(tmp_path, monkeypatch):
    async with pipeline(tmp_path) as (coordinator, _, _):
        commands = []
        observe_native_command(monkeypatch, coordinator, commands.append)
        client = Client(coordinator)

        def convert(output):
            rid, item = f"r{output.input_seq}", f"i{output.input_seq}"
            return [
                ResponseStarted(rid),
                TextDelta(rid, item, str(output.payload)),
                TextFinished(rid, item, str(output.payload)),
                ResponseFinished(
                    rid, item, str(output.payload), False, "completed", "stop"
                ),
            ]

        deployment = RealtimeDeployment(
            Capabilities(),
            lambda: CoordinatorAdapter(
                client,
                stages=["source", "sink"],
                request_builder=lambda cfg: OmniRequest(inputs=None),
                output_converter=convert,
                atomic_consumption=True,
            ),
        )
        async with endpoint(deployment=deployment) as (_, url, _, app):
            async with websockets.connect(url) as ws:
                created = await recv(ws)
                assert not native_sessions(coordinator)
                await send(ws, "session.update", session={})
                await recv(ws)
                assert created["session"]["id"] in native_sessions(coordinator)
                await append(ws, 0, 80)
                await recv(ws)
                await send(ws, "sglang.input_audio.end")
                drained, events = await until(ws, "sglang.input_audio.drained")
                assert drained["consumed_ms"] == 5
                assert any(e["type"] == "response.output_text.delta" for e in events)
                await send(ws, "session.close")
                assert (await recv(ws))["type"] == "session.closed"
                assert not native_sessions(coordinator)
                assert "close" in commands


@pytest.mark.asyncio
async def test_existing_rust_router_shared_handshake(tmp_path):
    binary = os.environ.get("OMNI_ROUTER_TEST_BINARY")
    if not binary:
        pytest.skip("set OMNI_ROUTER_TEST_BINARY to the built existing router")
    async with endpoint() as (http, _, producer, app):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        config = tmp_path / "router.toml"
        config.write_text(
            f"""schema_version = 1
[server]
listen = "127.0.0.1:{port}"
[shutdown]
drain_timeout_ms = 1000
[logging]
format = "json"
filter = "error"
[router]
strategy = "round_robin"
[admission]
global = 2
speech_websocket = 1
realtime_websocket = 2
[health]
interval_ms = 100
timeout_ms = 100
success_threshold = 1
failure_threshold = 1
[websocket.speech]
trust_domain = "local"
[websocket.realtime]
trust_domain = "local"
[[workers]]
worker_id = "mock"
base_url = "{http}"
trust_domain = "local"
default_model_id = "mock"
[workers.capacity]
speech_websocket = 1
realtime_websocket = 2
[[workers.service_profiles]]
service = "speech_websocket"
model_ids = ["mock"]
response_formats = ["pcm"]
stream_modes = ["non_streaming", "streaming"]
tasks = ["text_to_speech"]
reference_forms = ["none"]
voice_name_policy = "preset"
[[workers.service_profiles]]
service = "realtime_websocket"
"""
        )
        process = await asyncio.create_subprocess_exec(
            binary,
            "--config",
            str(config),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            async with httpx.AsyncClient() as client:
                async with asyncio.timeout(5):
                    while True:
                        assert process.returncode is None, (
                            (await process.stderr.read()).decode()
                            if process.returncode is not None
                            else ""
                        )
                        try:
                            if (
                                await client.get(f"http://127.0.0.1:{port}/ready")
                            ).is_success:
                                break
                        except httpx.TransportError:
                            pass
            async with websockets.connect(
                f"ws://127.0.0.1:{port}/v1/realtime?model=mock"
            ) as ws:
                assert (await recv(ws))["type"] == "session.created"
                assert producer.opened == 0
                await send(ws, "session.update", session={})
                assert (await recv(ws))["type"] == "session.updated"
                await append(ws, 0)
                assert (await recv(ws))["type"] == "sglang.input_audio.accepted"
                await send(ws, "session.close")
                assert (await recv(ws))["type"] == "session.closed"
        finally:
            if process.returncode is None:
                process.terminate()
            await asyncio.wait_for(process.wait(), 5)


@pytest.mark.asyncio
async def test_default_realtime_entry_uses_shared_ga_protocol(monkeypatch):
    monkeypatch.setattr(smart_turn, "load_smart_turn", lambda: None)
    async with endpoint(default=True) as (http, url, _, app):
        released = observe_connection_release(monkeypatch, app.state.realtime_manager)
        async with httpx.AsyncClient() as client:
            caps = (await client.get(http + "/v1/realtime/capabilities")).json()
        assert caps["native_full_duplex"] is False
        async with websockets.connect(url) as ws:
            created = await recv(ws)
            assert created["session"]["sglang"]["granted"] is None
            await send(
                ws,
                "session.update",
                session={
                    "audio": {"input": {"turn_detection": None}},
                    "output_modalities": ["text"],
                },
            )
            updated, _ = await until(ws, "session.updated")
            assert updated["session"]["audio"]["input"]["format"]["type"] == "audio/pcm"
            assert updated["session"]["sglang"]["granted"]["output_modalities"] == [
                "text"
            ]
            await send(ws, "session.close")
            await until(ws, "session.closed")
        await asyncio.wait_for(released.wait(), 5)
        assert not app.state.realtime_manager.active_sessions()
