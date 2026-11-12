# SPDX-License-Identifier: Apache-2.0
"""The real /v1/realtime route over a model-free adapter, with a lifecycle debug route."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI

from benchmarks.duplex.lifecycle_stress.host_metrics import (
    peak_rss_megabytes,
    read_open_fd_count,
    read_rss_megabytes,
)
from benchmarks.duplex.lifecycle_stress.records import ServerMode, ServerSnapshot
from sglang_omni.admission import QueueFullError
from sglang_omni.audio import smart_turn as smart_turn_module
from sglang_omni.audio.vad import Emit, VADConfig
from sglang_omni.client.client import Client
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.realtime import session as session_module
from sglang_omni.serve.realtime import turn_detector as turn_detector_module
from sglang_omni.serve.realtime.manager import RealtimeDeployment
from sglang_omni.serve.realtime.output import (
    ResponseFinished,
    ResponseStarted,
    TextDelta,
    TextFinished,
)
from sglang_omni.serve.realtime.schema import SessionConfiguration
from sglang_omni.serve.realtime.types import (
    Capabilities,
    InteractionAdapter,
    OutputSink,
    RuntimeLimits,
    Unit,
)

MOCK_MODEL_NAME = "lifecycle-stress-mock"
DEBUG_ROUTE_PATH = "/debug/lifecycle"
REALTIME_ROUTE_PATH = "/v1/realtime"
DEFAULT_REPLY_TEXT = "ok"
MS_PER_SECOND = 1000
STARTUP_POLL_INTERVAL_S = 0.01
STARTUP_TIMEOUT_S = 10.0
SHUTDOWN_TIMEOUT_S = 10.0


logger = logging.getLogger(__name__)


class HealthyCoordinator:
    """Stands in for the pipeline coordinator; the app only asks whether it is running."""

    def health(self) -> dict[str, bool]:
        return {"running": True}


class SilentVAD:
    """A turn detector that never hears speech, so silence stays silence without Silero."""

    def __init__(self, config: VADConfig | None = None) -> None:
        self.config = config

    def process(self, pcm_bytes: bytes) -> list[Emit]:
        return []

    def reset(self) -> None:
        pass


@dataclass(kw_only=True)
class ModelSlots:
    """How many sessions the stand-in model holds at once, shared by every adapter of one app."""

    limit: int
    in_use: int = 0


@dataclass(kw_only=True)
class EchoAdapter(InteractionAdapter):
    """Answers every native unit with a short text response so the server keeps sending."""

    reply_text: str = DEFAULT_REPLY_TEXT
    process_delay_ms: float = 0.0
    close_delay_ms: float = 0.0
    slots: ModelSlots | None = None
    output_sink: OutputSink | None = None
    unit_count: int = 0
    is_holding_slot: bool = False
    is_closed: bool = False

    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        # note (Yi Zhao): a model server whose queue is full rejects session.update this way and keeps the socket open.
        if self.slots is not None and self.slots.in_use >= self.slots.limit:
            raise QueueFullError()
        elif self.slots is not None:
            self.slots.in_use += 1
            self.is_holding_slot = True
        else:
            pass
        self.output_sink = emit

    async def process(self, unit: Unit) -> int:
        assert self.output_sink is not None, "process before open"
        if self.process_delay_ms > 0:
            await asyncio.sleep(self.process_delay_ms / MS_PER_SECOND)
        else:
            pass
        self.unit_count += 1
        response_id = f"response_{unit.index}"
        item_id = f"item_{unit.index}"
        await self.output_sink(ResponseStarted(response_id), unit)
        await self.output_sink(TextDelta(response_id, item_id, self.reply_text), unit)
        await self.output_sink(
            TextFinished(response_id, item_id, self.reply_text), unit
        )
        await self.output_sink(
            ResponseFinished(
                response_id, item_id, self.reply_text, False, "completed", "stop"
            ),
            unit,
        )
        return unit.real_samples

    async def close(self) -> None:
        if self.close_delay_ms > 0:
            await asyncio.sleep(self.close_delay_ms / MS_PER_SECOND)
        else:
            pass
        if self.is_holding_slot:
            assert self.slots is not None, "slot held without slots"
            self.slots.in_use -= 1
            self.is_holding_slot = False
        else:
            pass
        self.is_closed = True


@dataclass(kw_only=True)
class MockSettings:
    mode: ServerMode = "deployment"
    max_connections: int = 128
    model_slots: int | None = None
    idle_timeout_s: float = RuntimeLimits().idle_timeout_s
    session_update_timeout_s: float = RuntimeLimits().session_update_timeout_s
    cleanup_timeout_s: float = RuntimeLimits().cleanup_timeout_s
    reply_text: str = DEFAULT_REPLY_TEXT
    process_delay_ms: float = 0.0
    close_delay_ms: float = 0.0


def build_mock_app(settings: MockSettings) -> FastAPI:
    """The real app over fakes, plus a route for the session count and process resources a real server does not expose."""
    client = Client(HealthyCoordinator())
    if settings.mode == "deployment":
        slots = (
            None
            if settings.model_slots is None
            else ModelSlots(limit=settings.model_slots)
        )
        deployment = RealtimeDeployment(
            capabilities=Capabilities(),
            adapter_factory=lambda: EchoAdapter(
                reply_text=settings.reply_text,
                process_delay_ms=settings.process_delay_ms,
                close_delay_ms=settings.close_delay_ms,
                slots=slots,
            ),
            limits=RuntimeLimits(
                cleanup_timeout_s=settings.cleanup_timeout_s,
                session_update_timeout_s=settings.session_update_timeout_s,
                idle_timeout_s=settings.idle_timeout_s,
            ),
            max_connections=settings.max_connections,
        )
        app = create_app(
            client, model_name=MOCK_MODEL_NAME, realtime_deployment=deployment
        )
    else:
        # note (Yi Zhao): #3561 measures the per-connection Silero load; this benchmark measures the session lifecycle around it.
        session_module.StreamingVAD = SilentVAD
        turn_detector_module.StreamingVAD = SilentVAD
        smart_turn_module.load_smart_turn = lambda: None
        app = create_app(client, model_name=MOCK_MODEL_NAME, enable_realtime=True)

    @app.get(DEBUG_ROUTE_PATH, response_model=ServerSnapshot)
    async def lifecycle_snapshot() -> ServerSnapshot:
        return ServerSnapshot(
            taken_s=time.perf_counter(),
            mode=settings.mode,
            active_sessions=len(app.state.realtime_manager.sessions),
            rss_megabytes=read_rss_megabytes(os.getpid()),
            peak_rss_megabytes=peak_rss_megabytes(),
            task_count=len(asyncio.all_tasks()),
            thread_count=threading.active_count(),
            open_fd_count=read_open_fd_count(os.getpid()),
        )

    return app


@asynccontextmanager
async def serve_in_process(
    settings: MockSettings, host: str, port: int
) -> AsyncIterator[str]:
    """Serve the mock app on this event loop and yield its realtime URL."""
    config = uvicorn.Config(
        build_mock_app(settings),
        host=host,
        port=port,
        log_level="warning",
        ws_ping_interval=None,
        ws_ping_timeout=None,
    )
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())
    startup_deadline_s = time.monotonic() + STARTUP_TIMEOUT_S
    try:
        while not server.started:
            if serve_task.done():
                raise RuntimeError("uvicorn exited during startup")
            elif time.monotonic() >= startup_deadline_s:
                raise RuntimeError("uvicorn startup timed out")
            else:
                await asyncio.sleep(STARTUP_POLL_INTERVAL_S)
        bound_port = server.servers[0].sockets[0].getsockname()[1]
        yield f"ws://{host}:{bound_port}{REALTIME_ROUTE_PATH}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(serve_task, SHUTDOWN_TIMEOUT_S)


async def serve_mock(settings: MockSettings, host: str, port: int) -> None:
    """Serve until SIGINT or SIGTERM; prints the URL and pid the client needs."""
    # note (Yi Zhao): every aborted client socket makes asyncio warn once; the drop scenario aborts thousands.
    logging.getLogger("asyncio").setLevel(logging.ERROR)
    stop = asyncio.Event()
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(signal_number, stop.set)
    async with serve_in_process(settings, host, port) as url:
        print(f"Mock server listening on {url} pid={os.getpid()}", flush=True)
        await stop.wait()
