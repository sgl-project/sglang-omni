# SPDX-License-Identifier: Apache-2.0
"""Post-warmup GC freeze contracts of the API and stage processes."""

import asyncio
import gc
import queue
import socket
import subprocess
import sys
import weakref
from textwrap import dedent
from typing import Literal

import pytest

import sglang_omni.serve.launcher as launcher
from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.tp_control import TPLeaderFanout
from sglang_omni.proto import FreezeGCMessage, ShutdownMessage
from tests.unit_test.fixtures.pipeline_fakes import (
    RecordingCoordinatorControlPlane,
    RecordingStageControlPlane,
)
from tests.unit_test.pipeline.helpers import make_stage


class Cycle:
    def __init__(self) -> None:
        self.self_reference = self


@pytest.mark.parametrize("role", ["single", "leader"])
def test_stage_freezes_its_process_heap_on_the_freeze_message(role: str) -> None:
    """Every full collection in a warmed-up stage process rescanned the heap it built before serving."""
    warm = Cycle()
    warm_reference = weakref.ref(warm)
    follower_queue: queue.Queue = queue.Queue()
    control_plane = RecordingStageControlPlane()
    stage = make_stage(
        role=role,
        control_plane=control_plane,
        tp_fanout=(
            TPLeaderFanout(
                "stage", follower_work_queues=[follower_queue], follower_abort_queues=[]
            )
            if role == "leader"
            else None
        ),
    )

    async def run_until_shutdown() -> None:
        run_task = asyncio.create_task(stage.run())
        control_plane.inbox.put_nowait(
            deserialize_message(serialize_message(FreezeGCMessage()))
        )
        control_plane.inbox.put_nowait(ShutdownMessage())
        await asyncio.wait_for(run_task, timeout=5)

    try:
        asyncio.run(run_until_shutdown())
        del warm
        serving = Cycle()
        serving_reference = weakref.ref(serving)
        del serving
        gc.collect()

        assert warm_reference() is not None
        assert serving_reference() is None
        assert [follower_queue.get_nowait() for _ in range(follower_queue.qsize())] == (
            [FreezeGCMessage(), ShutdownMessage()] if role == "leader" else []
        )
    finally:
        gc.unfreeze()


def test_coordinator_sends_the_freeze_to_every_stage_instance() -> None:
    coordinator = Coordinator(
        "inproc://complete", "inproc://abort", entry_stage="preprocessing"
    )
    control_plane = RecordingCoordinatorControlPlane()
    coordinator.control_plane = control_plane
    stage_names = ("preprocessing@r0", "preprocessing@r1", "thinker")
    for stage_name in stage_names:
        coordinator.register_stage(stage_name, f"inproc://{stage_name}")

    asyncio.run(coordinator.freeze_gc())

    assert control_plane.submitted == [
        (stage_name, f"inproc://{stage_name}", FreezeGCMessage())
        for stage_name in stage_names
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("warmed", [False, True])
@pytest.mark.parametrize("serve_outcome", ["complete", "error", "cancelled"])
async def test_api_gc_is_unfrozen_on_exit_only_when_frozen(
    monkeypatch: pytest.MonkeyPatch,
    warmed: bool,
    serve_outcome: Literal["complete", "error", "cancelled"],
) -> None:
    events: list[str] = []

    async def serve(
        server: launcher.uvicorn.Server,
        sockets: list[socket.socket] | None = None,
    ) -> None:
        events.append("serve")
        if warmed:
            server.freeze_api_gc()
        else:
            pass
        if serve_outcome == "error":
            raise RuntimeError("serve failed")
        elif serve_outcome == "cancelled":
            raise asyncio.CancelledError
        else:
            pass

    monkeypatch.setattr(launcher.uvicorn.Server, "serve", serve)
    monkeypatch.setattr(launcher.gc, "get_freeze_count", lambda: 23)
    monkeypatch.setattr(launcher.gc, "collect", lambda: events.append("collect"))
    monkeypatch.setattr(launcher.gc, "freeze", lambda: events.append("freeze"))
    monkeypatch.setattr(launcher.gc, "unfreeze", lambda: events.append("unfreeze"))
    server = launcher.PipelineUvicornServer(launcher.uvicorn.Config("unused:app"))
    if serve_outcome == "complete":
        await server.serve()
    else:
        exception = RuntimeError if serve_outcome == "error" else asyncio.CancelledError
        with pytest.raises(exception):
            await server.serve()
    assert events == (["serve", "freeze", "unfreeze"] if warmed else ["serve"])
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
            server.freeze_api_gc()
            assert gc.get_freeze_count() > 0
            del startup

            request = Cycle()
            request.self_ref = request
            reference = weakref.ref(request)
            del request
            gc.collect()
            assert reference() is None
            assert startup_reference() is not None

        launcher.uvicorn.Server.serve = serve
        if sys.argv[1] == 'False':
            gc.disable()
        else:
            gc.enable()
        was_enabled = gc.isenabled()
        thresholds = gc.get_threshold()
        server = launcher.PipelineUvicornServer(launcher.uvicorn.Config('unused:app'))
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
