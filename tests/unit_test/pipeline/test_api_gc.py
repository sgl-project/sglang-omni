# SPDX-License-Identifier: Apache-2.0
"""API startup GC scope and request-cycle collection contracts."""

import asyncio
import socket
import subprocess
import sys
from textwrap import dedent
from typing import Literal

import pytest

import sglang_omni.serve.launcher as launcher
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.models.qwen3_asr.config import Qwen3ASRPipelineConfig


def test_only_qwen3_asr_enables_api_startup_freeze() -> None:
    assert PipelineConfig.freeze_api_gc_on_startup is False
    assert Qwen3ASRPipelineConfig.freeze_api_gc_on_startup is True
    assert "freeze_api_gc_on_startup" not in PipelineConfig.model_fields


@pytest.mark.asyncio
@pytest.mark.parametrize("freeze_enabled", [False, True])
@pytest.mark.parametrize("ready", [False, True])
async def test_api_freeze_waits_for_successful_startup(
    monkeypatch: pytest.MonkeyPatch, freeze_enabled: bool, ready: bool
) -> None:
    events: list[str] = []

    async def startup(
        server: launcher.uvicorn.Server,
        sockets: list[socket.socket] | None = None,
    ) -> None:
        events.append("startup")
        server.started = ready

    monkeypatch.setattr(launcher.uvicorn.Server, "startup", startup)
    monkeypatch.setattr(launcher.gc, "collect", lambda: events.append("collect"))
    monkeypatch.setattr(launcher.gc, "freeze", lambda: events.append("freeze"))
    monkeypatch.setattr(launcher.gc, "get_freeze_count", lambda: 0)
    server = launcher.PipelineUvicornServer(
        launcher.uvicorn.Config("unused:app"),
        freeze_api_gc_on_startup=freeze_enabled,
    )
    await server.startup()
    assert events == (
        ["startup", "collect", "freeze"] if ready and freeze_enabled else ["startup"]
    )
    assert server.has_frozen_api_gc is (ready and freeze_enabled)


@pytest.mark.asyncio
@pytest.mark.parametrize("freeze_enabled", [False, True])
@pytest.mark.parametrize("serve_outcome", ["complete", "error", "cancelled"])
async def test_api_gc_is_unfrozen_on_exit_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    freeze_enabled: bool,
    serve_outcome: Literal["complete", "error", "cancelled"],
) -> None:
    events: list[str] = []

    async def startup(
        server: launcher.uvicorn.Server,
        sockets: list[socket.socket] | None = None,
    ) -> None:
        events.append("startup")
        server.started = True

    async def serve(
        server: launcher.uvicorn.Server,
        sockets: list[socket.socket] | None = None,
    ) -> None:
        events.append("serve")
        await server.startup(sockets=sockets)
        if serve_outcome == "error":
            raise RuntimeError("serve failed")
        elif serve_outcome == "cancelled":
            raise asyncio.CancelledError
        else:
            pass

    monkeypatch.setattr(launcher.uvicorn.Server, "startup", startup)
    monkeypatch.setattr(launcher.uvicorn.Server, "serve", serve)
    monkeypatch.setattr(launcher.gc, "get_freeze_count", lambda: 23)
    monkeypatch.setattr(launcher.gc, "collect", lambda: events.append("collect"))
    monkeypatch.setattr(launcher.gc, "freeze", lambda: events.append("freeze"))
    monkeypatch.setattr(launcher.gc, "unfreeze", lambda: events.append("unfreeze"))
    server = launcher.PipelineUvicornServer(
        launcher.uvicorn.Config("unused:app"), freeze_api_gc_on_startup=freeze_enabled
    )
    if serve_outcome == "complete":
        await server.serve()
    else:
        exception = RuntimeError if serve_outcome == "error" else asyncio.CancelledError
        with pytest.raises(exception):
            await server.serve()
    assert events == (
        ["serve", "startup", "collect", "freeze", "unfreeze"]
        if freeze_enabled
        else ["serve", "startup"]
    )
    assert server.has_frozen_api_gc is False


@pytest.mark.parametrize("automatic_gc", [False, True])
@pytest.mark.parametrize("has_existing_frozen_graph", [False, True])
def test_request_cycles_and_shutdown_cycles_are_collectable(
    automatic_gc: bool, has_existing_frozen_graph: bool
) -> None:
    script = dedent(
        """
        import asyncio
        import gc
        import socket
        import sys
        import weakref
        from sglang_omni.serve import launcher

        if sys.argv[2] == 'True':
            gc.freeze()
        else:
            pass

        async def ready(
            server: launcher.uvicorn.Server,
            sockets: list[socket.socket] | None = None,
        ) -> None:
            server.started = True

        class Cycle:
            self_ref: "Cycle"

        startup = Cycle()
        startup.self_ref = startup
        startup_reference = weakref.ref(startup)

        async def serve(
            server: launcher.uvicorn.Server,
            sockets: list[socket.socket] | None = None,
        ) -> None:
            global startup
            await server.startup(sockets=sockets)
            assert gc.get_freeze_count() > 0
            del startup

            request = Cycle()
            request.self_ref = request
            reference = weakref.ref(request)
            del request
            gc.collect()
            assert reference() is None
            assert startup_reference() is not None

        launcher.uvicorn.Server.startup = ready
        launcher.uvicorn.Server.serve = serve
        if sys.argv[1] == 'False':
            gc.disable()
        else:
            gc.enable()
        was_enabled = gc.isenabled()
        thresholds = gc.get_threshold()
        server = launcher.PipelineUvicornServer(
            launcher.uvicorn.Config('unused:app'),
            freeze_api_gc_on_startup=True,
        )
        asyncio.run(server.serve())
        assert gc.get_freeze_count() == 0
        gc.collect()
        assert startup_reference() is None
        assert gc.isenabled() == was_enabled
        assert gc.get_threshold() == thresholds

        print('request_and_shutdown_cycles_collected')
        """
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(automatic_gc),
            str(has_existing_frozen_graph),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "request_and_shutdown_cycles_collected" in result.stdout
