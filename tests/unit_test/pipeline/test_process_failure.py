# SPDX-License-Identifier: Apache-2.0
"""A live worker must not hide a fatal error behind blocked cleanup."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import threading
from unittest.mock import AsyncMock, Mock

import pytest

from sglang_omni.pipeline import stage_workers
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.pipeline.stage_workers import (
    StageGroup,
    StageLaunchConfig,
    StageWorkerProcessSpec,
)
from tests.unit_test.pipeline.helpers import make_stage


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
