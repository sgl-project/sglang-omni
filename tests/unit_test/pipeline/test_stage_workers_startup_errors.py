# SPDX-License-Identifier: Apache-2.0
"""Deterministic checks for how wait_ready reports a child startup failure.

These use fake processes and fake channels instead of spawning children, so
they exercise the ordering and the read mode directly and do not depend on how
long a real child takes to start.
"""

from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

from sglang_omni.pipeline.stage_workers import StageGroup

TRACEBACK = "Traceback (most recent call last):\n  RuntimeError: factory boom"


class _DeadProcess:
    exitcode = 1

    @staticmethod
    def is_alive() -> bool:
        return False


class _LiveProcess:
    exitcode = None

    @staticmethod
    def is_alive() -> bool:
        return True


class _NeverReady:
    @staticmethod
    def is_set() -> bool:
        return False

    @staticmethod
    def wait(timeout: float | None = None) -> bool:
        del timeout
        return False


class _InFlightChannel:
    """A queue whose payload has not reached the parent's buffer yet.

    `multiprocessing.Queue.put` hands the bytes to a feeder thread, so a child
    can exit before a non-blocking read on the parent can see them.
    """

    def __init__(self, payload: str) -> None:
        self._payload = payload

    def get_nowait(self) -> str:
        raise queue.Empty

    def get(self, timeout: float | None = None) -> str:
        del timeout
        return self._payload


class _ReadyChannel:
    def __init__(self, payload: str) -> None:
        self._payload = payload
        self._taken = False

    def get_nowait(self) -> str:
        if self._taken:
            raise queue.Empty
        self._taken = True
        return self._payload

    def get(self, timeout: float | None = None) -> str:
        del timeout
        raise queue.Empty


class _EmptyChannel:
    @staticmethod
    def get_nowait() -> str:
        raise queue.Empty

    @staticmethod
    def get(timeout: float | None = None) -> str:
        del timeout
        raise queue.Empty


def _group(proc, channel) -> StageGroup:
    group = StageGroup("g", [SimpleNamespace(process_name="worker")])
    group._processes = [proc]
    group._ready_events = [_NeverReady()]
    group._startup_error_channels = [channel]
    return group


@pytest.mark.asyncio
async def test_dead_child_reports_a_traceback_still_in_flight() -> None:
    """The read must not be non-blocking on the death path.

    With `get_nowait()` here the parent reports only an exit code, which is the
    regression this covers.
    """
    group = _group(_DeadProcess(), _InFlightChannel(TRACEBACK))

    with pytest.raises(RuntimeError, match="factory boom") as excinfo:
        await group.wait_ready(timeout=5.0)

    assert "died during startup" in str(excinfo.value)
    assert "exit code 1" in str(excinfo.value)


@pytest.mark.asyncio
async def test_live_child_that_already_reported_is_not_left_to_time_out() -> None:
    group = _group(_LiveProcess(), _ReadyChannel(TRACEBACK))

    with pytest.raises(RuntimeError, match="factory boom") as excinfo:
        await group.wait_ready(timeout=5.0)

    assert "failed during startup" in str(excinfo.value)


@pytest.mark.asyncio
async def test_dead_child_without_a_traceback_still_names_the_exit_code() -> None:
    group = _group(_DeadProcess(), _EmptyChannel())

    with pytest.raises(RuntimeError, match="died during startup") as excinfo:
        await group.wait_ready(timeout=5.0)

    assert "exit code 1" in str(excinfo.value)
    assert "Startup failure detail" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_live_silent_child_times_out() -> None:
    group = _group(_LiveProcess(), _EmptyChannel())

    with pytest.raises(TimeoutError, match="did not become ready"):
        await group.wait_ready(timeout=0.3)
