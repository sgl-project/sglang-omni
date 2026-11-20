# SPDX-License-Identifier: Apache-2.0
"""Tests of session closure through a WebSocket connection."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest
import uvicorn

from sglang_omni.client.client import Client
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.realtime.manager import (
    RealtimeDeployment,
    RealtimeSessionManager,
)
from sglang_omni.serve.realtime.types import Capabilities, RuntimeLimits
from tests.unit_test.fixtures.realtime_fakes import (
    MODEL_NAME,
    HealthCoordinator,
    ScriptedAdapter,
)

# Note (Haiyang Luo): The asyncio client API is available in websockets 13 and later.
pytest.importorskip("websockets", minversion="13.0")

from websockets.asyncio.client import (  # noqa: E402 - websockets version gate
    ClientConnection,
    connect,
)

PING_INTERVAL_SECONDS = 0.2
ADMISSION_TIMEOUT_SECONDS = 1.2
# note (Haiyang Luo): The client answers several pings inside one idle window.
IDLE_TIMEOUT_SECONDS = 3 * PING_INTERVAL_SECONDS
RELEASE_MARGIN_SECONDS = 2.0
SOCKET_WAIT_TIMEOUT_SECONDS = 5.0
STARTUP_POLL_INTERVAL_SECONDS = 0.01
RELEASE_POLL_INTERVAL_SECONDS = 0.02


@dataclass(kw_only=True)
class LiveServer:
    url: str
    adapter: ScriptedAdapter
    session_manager: RealtimeSessionManager


@asynccontextmanager
async def serve_live(limits: RuntimeLimits) -> AsyncIterator[LiveServer]:
    adapter = ScriptedAdapter()
    deployment = RealtimeDeployment(
        capabilities=Capabilities(),
        adapter_factory=lambda: adapter,
        limits=limits,
        max_connections=1,
    )
    app = create_app(
        Client(HealthCoordinator(True)),
        model_name=MODEL_NAME,
        realtime_deployment=deployment,
    )
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        log_level="warning",
        ws_ping_interval=PING_INTERVAL_SECONDS,
        ws_ping_timeout=SOCKET_WAIT_TIMEOUT_SECONDS,
    )
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())
    startup_deadline_seconds = time.monotonic() + SOCKET_WAIT_TIMEOUT_SECONDS
    try:
        while not server.started:
            assert not serve_task.done(), "uvicorn exited during startup"
            assert (
                time.monotonic() < startup_deadline_seconds
            ), "uvicorn startup timed out"
            await asyncio.sleep(STARTUP_POLL_INTERVAL_SECONDS)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield LiveServer(
            url=f"ws://127.0.0.1:{port}/v1/realtime",
            adapter=adapter,
            session_manager=app.state.realtime_manager,
        )
    finally:
        server.should_exit = True
        await asyncio.wait_for(serve_task, SOCKET_WAIT_TIMEOUT_SECONDS)


async def receive_until(connection: ClientConnection, event_type: str) -> None:
    deadline_seconds = time.monotonic() + SOCKET_WAIT_TIMEOUT_SECONDS
    while True:
        message = await asyncio.wait_for(
            connection.recv(), deadline_seconds - time.monotonic()
        )
        if json.loads(message)["type"] == event_type:
            return
        else:
            pass


async def open_session(connection: ClientConnection) -> None:
    await receive_until(connection, "session.created")
    await connection.send(
        json.dumps(
            {"event_id": "client_update", "type": "session.update", "session": {}}
        )
    )
    await receive_until(connection, "session.updated")


async def assert_released(live_server: LiveServer, timeout_seconds: float) -> None:
    deadline_seconds = time.monotonic() + timeout_seconds
    while live_server.session_manager.sessions:
        assert time.monotonic() < deadline_seconds, "session was not released"
        await asyncio.sleep(RELEASE_POLL_INTERVAL_SECONDS)
    async with connect(live_server.url, ping_interval=None) as connection:
        await receive_until(connection, "session.created")


@pytest.mark.asyncio
async def test_client_close_releases_session() -> None:
    """Client-requested closure must include client_closed in the WebSocket close frame."""
    async with serve_live(RuntimeLimits()) as live_server:
        async with connect(live_server.url, ping_interval=None) as connection:
            await open_session(connection)
            await connection.send(
                json.dumps({"event_id": "client_close", "type": "session.close"})
            )
            await receive_until(connection, "session.closed")
        assert (connection.close_code, connection.close_reason) == (
            1000,
            "client_closed",
        )
        await assert_released(live_server, RELEASE_MARGIN_SECONDS)
        assert live_server.adapter.is_closed


@pytest.mark.asyncio
async def test_session_update_timeout_reports_policy_close_over_socket() -> None:
    """An initial session.update timeout must close the socket with 1008 policy violation."""
    limits = RuntimeLimits(session_update_timeout_s=ADMISSION_TIMEOUT_SECONDS)
    async with serve_live(limits) as live_server:
        async with connect(live_server.url, ping_interval=None) as connection:
            await receive_until(connection, "session.created")
            await asyncio.wait_for(
                connection.wait_closed(),
                ADMISSION_TIMEOUT_SECONDS + RELEASE_MARGIN_SECONDS,
            )

        assert (connection.close_code, connection.close_reason) == (
            1008,
            "session_update_timeout",
        )
        await assert_released(live_server, RELEASE_MARGIN_SECONDS)
        assert not live_server.adapter.is_closed


@pytest.mark.asyncio
async def test_open_peer_answering_pings_is_closed_by_idle_timeout() -> None:
    limits = RuntimeLimits(idle_timeout_s=IDLE_TIMEOUT_SECONDS)
    async with serve_live(limits) as live_server:
        async with connect(live_server.url, ping_interval=None) as connection:
            started_at_seconds = time.monotonic()
            await open_session(connection)
            await asyncio.wait_for(
                connection.wait_closed(),
                IDLE_TIMEOUT_SECONDS + RELEASE_MARGIN_SECONDS,
            )
            closed_after_seconds = time.monotonic() - started_at_seconds

        assert closed_after_seconds >= IDLE_TIMEOUT_SECONDS
        assert (connection.close_code, connection.close_reason) == (
            1008,
            "idle_timeout",
        )
        await assert_released(live_server, RELEASE_MARGIN_SECONDS)
        assert live_server.adapter.is_closed
