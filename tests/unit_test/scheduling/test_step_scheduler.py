# SPDX-License-Identifier: Apache-2.0
"""Behavioral coverage for StepScheduler request ownership."""

import gc
import subprocess
import sys
import threading
import time
import weakref
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from sglang_omni.admission import QueueFullError
from sglang_omni.scheduling import step_scheduler
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.step_scheduler import (
    ShutdownCallback,
    StepResult,
    StepScheduler,
)


def scheduler_for(
    *,
    max_workers: int = 1,
    max_running_requests: int = 16,
    max_queued_requests: int = 64,
    shutdown_callback: ShutdownCallback | None = None,
) -> StepScheduler:
    return StepScheduler(
        lambda task: task,
        max_workers=max_workers,
        max_running_requests=max_running_requests,
        max_queued_requests=max_queued_requests,
        shutdown_callback=shutdown_callback or (lambda: None),
    )


@contextmanager
def running_scheduler(
    *,
    max_workers: int = 1,
    max_running_requests: int = 16,
    max_queued_requests: int = 64,
) -> Iterator[StepScheduler]:
    scheduler = scheduler_for(
        max_workers=max_workers,
        max_running_requests=max_running_requests,
        max_queued_requests=max_queued_requests,
    )
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    try:
        yield scheduler
    finally:
        scheduler.stop()
        thread.join(timeout=5)
        assert not thread.is_alive()


class Task:
    def __init__(
        self,
        name: str,
        steps: int | None,
        trace: list[str],
        *,
        release: threading.Event | None = None,
        bounded: bool = False,
    ) -> None:
        self.name = name
        self.remaining = steps
        self.trace = trace
        self.entered = threading.Event()
        self.release = release
        self.bounded_by_stage = bounded
        self.closed: list[bool] = []
        self.finished = threading.Event()

    def step(self) -> StepResult:
        self.trace.append(self.name)
        self.entered.set()
        if self.release is not None:
            assert self.release.wait(timeout=5)
        else:
            pass
        if self.remaining is None:
            raise ValueError("model step failed")
        else:
            self.remaining -= 1
            return StepResult(self.remaining == 0, self.name)

    def close(self, *, aborted: bool) -> None:
        self.closed.append(aborted)
        self.finished.set()


def submit(scheduler: StepScheduler, task: Task) -> None:
    scheduler.inbox.put(IncomingMessage(task.name, "new_request", task))


def test_first_steps_precede_continuing_work_and_uploads_finish_in_order() -> None:
    trace, release = [], threading.Event()
    first = Task("first", 3, trace, release=release)
    second, live = Task("second", 3, trace), Task("live", 1, trace)
    with running_scheduler() as scheduler:
        submit(scheduler, first)
        assert first.entered.wait(timeout=5)
        submit(scheduler, second)
        submit(scheduler, live)
        release.set()
        outputs = [scheduler.outbox.get(timeout=5).data for _ in range(3)]
    assert outputs == ["live", "first", "second"]
    assert first.closed == second.closed == live.closed == [False]
    assert trace == ["first", "second", "live", "first", "first", "second", "second"]


@pytest.mark.parametrize("workers", [2])
def test_abort_waits_for_inflight_work_and_preserves_other_requests(
    workers: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(step_scheduler, "ABORTED_REQUEST_ID_LIMIT", 1)
    trace, release = [], threading.Event()
    cancelled = Task("cancelled", 3, trace, release=release)
    healthy = Task("healthy", 1, trace)
    with running_scheduler(max_workers=workers) as scheduler:
        submit(scheduler, cancelled)
        assert cancelled.entered.wait(timeout=5)
        scheduler.abort("cancelled")
        scheduler.abort("unrelated")
        assert not cancelled.closed
        submit(scheduler, healthy)
        release.set()
        assert scheduler.outbox.get(timeout=5).request_id == "healthy"
    assert trace.count("cancelled") == 1
    assert cancelled.closed == [True]
    assert healthy.closed == [False]


def test_queue_limit_rejects_excess_except_stage_bounded_work_and_shutdown_closes() -> (
    None
):
    trace, release = [], threading.Event()
    active = Task("active", 3, trace, release=release)
    queued, excess = Task("queued", 1, trace), Task("excess", 1, trace)
    bounded = Task("bounded", 1, trace, bounded=True)
    with running_scheduler(max_running_requests=1, max_queued_requests=1) as scheduler:
        submit(scheduler, active)
        assert active.entered.wait(timeout=5)
        for task in (queued, bounded, excess):
            submit(scheduler, task)
        try:
            rejection = scheduler.outbox.get(timeout=5)
            assert rejection.request_id == "excess"
            assert isinstance(rejection.data, QueueFullError)
            scheduler.stop()
        finally:
            release.set()
    assert active.closed == queued.closed == bounded.closed == excess.closed == [True]
    assert trace == ["active"]


def test_failed_step_releases_capacity_and_next_request_succeeds() -> None:
    failed, healthy = Task("failed", None, []), Task("healthy", 1, [])
    with running_scheduler(max_running_requests=1) as scheduler:
        submit(scheduler, failed)
        submit(scheduler, healthy)
        failure = scheduler.outbox.get(timeout=5)
        success = scheduler.outbox.get(timeout=5)
    assert (failure.type, str(failure.data)) == ("error", "model step failed")
    assert (success.type, success.data) == ("result", "healthy")
    assert failed.closed == [True]
    assert healthy.closed == [False]


def test_cancelled_queued_requests_are_released_while_work_is_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RequestId(str):
        pass

    monkeypatch.setattr(step_scheduler, "ABORTED_REQUEST_ID_LIMIT", 1)
    release = threading.Event()
    active = Task("active", 1, [], release=release)
    references = []
    with running_scheduler(max_running_requests=1, max_queued_requests=1) as scheduler:
        submit(scheduler, active)
        try:
            assert active.entered.wait(timeout=5)
            for index in range(8):
                queued = Task(RequestId(f"cancelled-{index}"), 1, [])
                references.append(weakref.ref(queued.name))
                submit(scheduler, queued)
                deadline = time.monotonic() + 5
                while queued.name not in scheduler.requests:
                    assert time.monotonic() < deadline
                    time.sleep(0.001)
                scheduler.abort(queued.name)
                assert queued.finished.wait(timeout=5)
                assert queued.closed == [True]
                del queued
            gc.collect()
            assert all(reference() is None for reference in references[:-1])
        finally:
            release.set()


@pytest.mark.parametrize("stop_before_start", [False, True])
def test_stop_at_idle_boundary_exits_without_another_request(
    stop_before_start: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[bool] = []
    scheduler = scheduler_for(shutdown_callback=lambda: closed.append(True))
    clear = scheduler.wakeup.clear

    def stop_then_clear() -> None:
        scheduler.stop()
        clear()

    if stop_before_start:
        scheduler.stop()
    else:
        monkeypatch.setattr(scheduler.wakeup, "clear", stop_then_clear)
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    try:
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert closed == [True]
    finally:
        scheduler.stop()
        thread.join(timeout=5)


def test_step_scheduler_imports_no_model_runtime() -> None:
    """Stages that never allocate KV cache must not pay for the AR runtime."""
    script = (
        "import sys\n"
        "import sglang_omni.scheduling.step_scheduler\n"
        "loaded = sorted(\n"
        "    name for name in sys.modules\n"
        "    if name == 'torch' or name.startswith(('sglang.', 'torch.'))\n"
        ")\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True, timeout=120)
