# SPDX-License-Identifier: Apache-2.0
"""A live worker must not hide a fatal error behind blocked cleanup."""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import threading
from contextlib import nullcontext
from multiprocessing.synchronize import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from sglang_omni.client import Client
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.pipeline import stage_workers
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.pipeline.stage_workers import (
    StageGroup,
    StageLaunchConfig,
    StageWorkerProcessSpec,
)
from sglang_omni.proto import StreamMessage, SubmitMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.serve.openai_api import create_app
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane
from tests.unit_test.pipeline.helpers import make_stage, stage


@pytest.mark.asyncio
async def test_receive_failure_is_reported_before_blocked_stage_stop() -> None:
    """A receive failure used to stay invisible while Stage.stop was blocked."""
    failure_event = threading.Event()
    cleanup_entered = asyncio.Event()
    release_cleanup = asyncio.Event()
    stage = make_stage(process_failure_callback=failure_event.set)
    stage.start = AsyncMock()
    stage.running = True
    stage.control_plane.recv = AsyncMock(
        side_effect=RuntimeError("control receive failed")
    )

    async def stop() -> None:
        cleanup_entered.set()
        await release_cleanup.wait()

    stage.stop = stop
    running = asyncio.create_task(stage.run())
    try:
        await asyncio.wait_for(cleanup_entered.wait(), timeout=2)
        assert not running.done()
        assert failure_event.is_set()
    finally:
        release_cleanup.set()
        with pytest.raises(RuntimeError, match="control receive failed"):
            await asyncio.wait_for(running, timeout=2)


@pytest.mark.parametrize(
    ("partially_started", "error", "reported"),
    [
        (False, RuntimeError("startup failure"), True),
        (True, RuntimeError("startup failure"), True),
        (False, KeyboardInterrupt(), False),
        (False, asyncio.CancelledError(), False),
    ],
)
def test_process_failure_is_reported_before_outer_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    partially_started: bool,
    error: BaseException,
    reported: bool,
) -> None:
    """Partial startup used to enter Stage.stop before notifying the parent, and an orderly SIGINT stop used to read as a failure."""
    failure_event = multiprocessing.get_context("spawn").Event()
    startup_error_channel = multiprocessing.get_context("spawn").Queue()
    spec = StageWorkerProcessSpec(
        "worker", [StageLaunchConfig("worker")], failure_event=failure_event
    )
    constructed = Mock(spec=Stage)
    constructed.name = "worker"
    constructed.start = AsyncMock(side_effect=error)
    constructed.running = partially_started

    async def stop() -> None:
        assert failure_event.is_set() is reported
        constructed.running = False

    constructed.stop = AsyncMock(side_effect=stop)
    monkeypatch.setattr(
        stage_workers, "construct_stage", Mock(return_value=constructed)
    )
    cleanup_called = []

    def cleanup(stages: list[Stage], log: logging.Logger, *, reason: str) -> None:
        assert failure_event.is_set() is reported
        if reported:
            assert "startup failure" in startup_error_channel.get(timeout=1)
        else:
            assert startup_error_channel.empty()
        assert stages == [constructed]
        cleanup_called.append(reason)

    monkeypatch.setattr(stage_workers, "cleanup_constructed_stages", cleanup)
    try:
        with pytest.raises(type(error)):
            stage_workers.run_process(
                spec,
                multiprocessing.get_context("spawn").Event(),
                logging.getLogger(__name__),
                startup_error_channel,
            )
    finally:
        startup_error_channel.close()
        startup_error_channel.join_thread()
    assert cleanup_called == ["stage process worker failure"]
    assert constructed.stop.await_count == int(partially_started)


@pytest.mark.asyncio
async def test_ready_worker_failure_during_startup_does_not_wait_for_death() -> None:
    group = StageGroup(
        "worker", [StageWorkerProcessSpec("worker", [StageLaunchConfig("worker")])]
    )
    context = multiprocessing.get_context("spawn")
    failure_event = context.Event()
    failure_event.set()
    group.failure_events = [failure_event]
    group.ready_events = [context.Event()]
    group.ready_events[0].set()
    group.startup_error_channels = [context.Queue()]
    process = Mock()
    process.pid = 123
    process.is_alive.return_value = True
    group._processes = [
        process
    ]  # noqa: leading-underscore  # Existing worker lifecycle state has no public setter.
    try:
        with pytest.raises(RuntimeError, match="failure during startup: worker"):
            await group.wait_ready(timeout=1)
    finally:
        group.close_control_channels()


def run_failing_spec_scheduler(
    ready: Event, fail: Event, unsafe_cleanup: Event
) -> None:
    scheduler = object.__new__(OmniScheduler)
    scheduler.enable_overlap = True
    scheduler.enable_async_decode = True
    scheduler.stop_requested = False
    scheduler.shutdown_lock = threading.Lock()
    scheduler.model_runner = object()
    scheduler.session_bridge = None
    scheduler.gpu_id = 0
    scheduler.should_run_top_priority_stream = False
    scheduler.forward_stream = SimpleNamespace(cuda_stream=1)
    scheduler.copy_stream = SimpleNamespace(cuda_stream=2)
    scheduler.device_module = SimpleNamespace(
        set_device=lambda device: None,
        synchronize=lambda device: None,
        Stream=lambda **kwargs: SimpleNamespace(cuda_stream=3),
        stream=lambda stream: nullcontext(),
    )

    def forward(batch, proxy_tensors) -> None:
        ready.set()
        fail.wait()
        raise RuntimeError("injected speculative forward failure")

    scheduler._run_batch = (
        forward  # noqa: leading-underscore  # inject at the native launch boundary
    )
    scheduler.event_loop_async_decode = lambda: scheduler.run_batch(None)
    scheduler.handle_batch_failure = lambda *args: unsafe_cleanup.set()
    scheduler.emit_remaining_model_path_ends = lambda **kwargs: unsafe_cleanup.set()
    scheduler.shutdown_request_build_executor = unsafe_cleanup.set
    scheduler.discard_pending_request_admissions = unsafe_cleanup.set
    scheduler.shutdown_resources = unsafe_cleanup.set
    scheduler.start()


@pytest.mark.asyncio
async def test_worker_hard_exit_errors_http_requests_and_terminates_blocked_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker exit must finish both streaming and queued requests and reap its peer."""
    context = multiprocessing.get_context("spawn")
    ready_events = [context.Event(), context.Event()]
    fail_events = [context.Event(), context.Event()]
    unsafe_cleanup_events = [context.Event(), context.Event()]
    processes = [
        context.Process(target=run_failing_spec_scheduler, args=(ready, fail, cleanup))
        for ready, fail, cleanup in zip(
            ready_events, fail_events, unsafe_cleanup_events
        )
    ]
    group = StageGroup(
        "thinker",
        [
            StageWorkerProcessSpec(f"thinker-{rank}", [StageLaunchConfig("thinker")])
            for rank in range(2)
        ],
    )
    group._processes = (
        processes.copy()
    )  # noqa: leading-underscore  # real processes injected without loading models
    coordinator = Coordinator(
        completion_endpoint="inproc://complete",
        abort_endpoint="inproc://abort",
        entry_stage="thinker",
        terminal_stages=["thinker"],
    )
    control_plane = RecordingCoordinatorControlPlane()
    coordinator.control_plane = control_plane
    coordinator.register_stage("thinker", "inproc://thinker")
    coordinator.running = True
    submitted = asyncio.Event()
    request_ids: list[str] = []
    original_submit = control_plane.submit_to_stage

    async def submit(stage: str, endpoint: str, message: SubmitMessage) -> None:
        await original_submit(stage, endpoint, message)
        request_ids.append(message.request_id)
        if len(request_ids) == 3:
            submitted.set()

    monkeypatch.setattr(control_plane, "submit_to_stage", submit)
    original_shutdown = group.shutdown

    async def shutdown(before_signal=None) -> None:
        await original_shutdown(join_timeout=0, before_signal=before_signal)

    monkeypatch.setattr(group, "shutdown", shutdown)
    runner = MultiProcessPipelineRunner(
        PipelineConfig(model_path="dummy", stages=[stage("thinker", terminal=True)])
    )
    runner._coordinator = coordinator  # noqa: leading-underscore  # installed without starting model workers
    runner.groups = [group]
    runner.started = True
    runner.fatal_event = asyncio.Event()
    app = create_app(Client(coordinator), model_name="dummy")
    requests: list[asyncio.Task[httpx.Response]] = []
    try:
        for process in processes:
            process.start()
        for ready in ready_events:
            assert await asyncio.to_thread(ready.wait, 15)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            for streaming in (True, True, False):
                requests.append(
                    asyncio.create_task(
                        client.post(
                            "/v1/chat/completions",
                            json={
                                "model": "dummy",
                                "messages": [{"role": "user", "content": "hello"}],
                                "stream": streaming,
                            },
                        )
                    )
                )
            await asyncio.wait_for(submitted.wait(), 5)
            streaming_ids = [
                request_id
                for request_id in request_ids
                if request_id in coordinator.stream_queues
            ]
            await coordinator.handle_stream(
                StreamMessage(
                    request_id=streaming_ids[0],
                    from_stage="thinker",
                    stage_name="thinker",
                    modality="text",
                    chunk={"text": "partial", "modality": "text"},
                )
            )
            fail_events[0].set()
            await asyncio.to_thread(processes[0].join, 15)
            assert processes[0].exitcode == 1
            await asyncio.wait_for(runner.monitor_children(), 5)
            responses = await asyncio.wait_for(asyncio.gather(*requests), 5)

        stream_responses = responses[:2]
        assert sum("partial" in response.text for response in stream_responses) == 1
        for response in stream_responses:
            chunks = [
                line.removeprefix("data: ")
                for line in response.text.splitlines()
                if line.startswith("data: ")
            ]
            assert chunks[-1] == "[DONE]"
            error = json.loads(chunks[-2])["error"]
            assert "Dead stage process" in error["message"]
            assert response.status_code == 200
        assert responses[2].status_code == 500
        assert "Dead stage process" in responses[2].text
        assert runner.fatal_event.is_set()
        assert not coordinator.requests
        assert all(not process.is_alive() for process in processes)
        assert all(not cleanup.is_set() for cleanup in unsafe_cleanup_events)
    finally:
        for request in requests:
            if not request.done():
                request.cancel()
        await asyncio.gather(*requests, return_exceptions=True)
        for process in processes:
            if process.pid is not None and process.is_alive():
                process.kill()
                process.join(5)
        await runner.stop()
