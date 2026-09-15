# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import multiprocessing

import pytest

from sglang_omni.pipeline import stage_workers
from sglang_omni.pipeline.stage_workers import (
    StageGroup,
    StageLaunchConfig,
    StageWorkerProcessSpec,
)


def _signal_ready(event):
    event.set()


def _own_child(spec, ready_event, error_channel):
    child = multiprocessing.get_context("spawn").Process(
        target=_signal_ready, args=(ready_event,)
    )
    try:
        child.start()
    except AssertionError as exc:
        error_channel.put(str(exc))
        return
    try:
        child.join(timeout=10)
        if child.is_alive():
            raise RuntimeError("Owned child did not exit")
        if child.exitcode:
            raise RuntimeError(f"Owned child failed with code {child.exitcode}")
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)
        child.close()


@pytest.mark.parametrize("allow_children", [False, True])
def test_stage_process_child_ownership(monkeypatch, allow_children):
    monkeypatch.setattr(stage_workers, "stage_process_main", _own_child)
    spec = StageLaunchConfig(
        stage_name="native",
        allow_child_processes=allow_children,
    )
    group = StageGroup("native", [StageWorkerProcessSpec("native", [spec])])
    try:
        group.spawn(multiprocessing.get_context("spawn"))
        process = group.processes[0]
        assert process.daemon is not allow_children
        process.join(timeout=15)
        assert not process.is_alive()
        assert process.exitcode == 0
        assert group._ready_events[0].is_set() is allow_children
        if not allow_children:
            assert "daemonic processes are not allowed to have children" in (
                group._startup_error_channels[0].get(timeout=2)
            )
    finally:
        asyncio.run(group.shutdown(join_timeout=1))
