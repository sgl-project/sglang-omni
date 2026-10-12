# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from multiprocessing.util import Finalize
from pathlib import Path

import pytest

from sglang_omni.config import EndpointsConfig, PipelineConfig, StageConfig
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def resource_factory(resource_path: str, finalizer_path: str, block: bool = False):
    resource = Path(resource_path)
    finalizer = Path(finalizer_path)
    resource.touch()
    finalizer.touch()
    Finalize(None, finalizer.unlink, exitpriority=0)
    if block:
        Finalize(None, resource.unlink, exitpriority=0)
        while True:
            time.sleep(0.02)
    return SimpleScheduler(lambda payload: payload, shutdown_callback=resource.unlink)


def deferred_failure_factory(trigger_path: str):
    deadline = time.monotonic() + 40
    while not Path(trigger_path).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Failure trigger did not arrive")
        time.sleep(0.02)
    raise RuntimeError("Controlled later stage startup failure")


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("block", [False, True])
async def test_stage_releases_resources_on_partial_startup(tmp_path, cancel, block):
    resource = tmp_path / "scheduler-resource"
    finalizer = tmp_path / "process-resource"
    trigger = tmp_path / "fail"
    ipc = tmp_path / "ipc"
    runner = MultiProcessPipelineRunner(
        PipelineConfig(
            model_path="unused",
            name="rollback",
            entry_stage="ready",
            stages=[
                StageConfig(
                    name="ready",
                    process="ready",
                    factory_path=f"{__name__}.resource_factory",
                    factory={
                        "resource_path": str(resource),
                        "finalizer_path": str(finalizer),
                        "block": block,
                    },
                    terminal=True,
                ),
                StageConfig(
                    name="failing",
                    process="failing",
                    factory_path=f"{__name__}.deferred_failure_factory",
                    factory={"trigger_path": str(trigger)},
                    terminal=True,
                ),
            ],
            endpoints=EndpointsConfig(base_path=str(ipc)),
        )
    )
    starting = asyncio.create_task(runner.start(timeout=40))
    processes = []
    try:
        async with asyncio.timeout(35):
            while not (
                finalizer.exists()
                if block
                else runner.groups and runner.groups[0].is_ready
            ):
                if starting.done():
                    starting.result()
                await asyncio.sleep(0.02)
        processes = [p for group in runner.groups for p in group.processes]
        assert resource.exists() and finalizer.exists()
        if cancel:
            starting.cancel()
            expected = asyncio.CancelledError
        else:
            trigger.touch()
            expected = RuntimeError
        with pytest.raises(expected):
            await asyncio.wait_for(starting, timeout=15)
        assert not resource.exists()
        assert not finalizer.exists()
        assert all(not process.is_alive() for process in processes)
        assert not list(ipc.iterdir())
    finally:
        if not starting.done():
            starting.cancel()
        await asyncio.gather(starting, return_exceptions=True)
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
