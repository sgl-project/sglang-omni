# SPDX-License-Identifier: Apache-2.0
"""Behavioral coverage for StepScheduler request ownership."""

import gc
import subprocess
import sys
import threading
import time
import weakref
from contextlib import contextmanager

import pytest

from sglang_omni.admission import QueueFullError
from sglang_omni.proto.admin import ADMIN_MODEL_INFO, ADMIN_PAUSE_GENERATION
from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.step_scheduler import StepResult, StepScheduler


@contextmanager
def running_scheduler(builder, **kwargs):
    scheduler = StepScheduler(builder, **kwargs)
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    try:
        yield scheduler
    finally:
        scheduler.stop()
        thread.join(timeout=5)
        assert not thread.is_alive()


class Task:
    def __init__(self, name, steps, trace, *, entered=None, release=None, fail=False):
        self.name = name
        self.remaining = steps
        self.trace = trace
        self.entered = entered
        self.release = release
        self.fail = fail
        self.closed = []

    def step(self):
        self.trace.append(self.name)
        if self.entered is not None:
            self.entered.set()
            assert self.release.wait(timeout=5)
        if self.fail:
            raise ValueError("model step failed")
        self.remaining -= 1
        return StepResult(self.remaining == 0, self.name)

    def close(self, *, aborted):
        self.closed.append(aborted)


def submit(scheduler, task):
    scheduler.inbox.put(IncomingMessage(task.name, "new_request", task))


def test_first_steps_precede_continuing_work_and_uploads_finish_in_order():
    trace = []
    entered, release = threading.Event(), threading.Event()
    first = Task("first", 3, trace, entered=entered, release=release)
    second = Task("second", 3, trace)
    live = Task("live", 1, trace)
    with running_scheduler(lambda item: item) as scheduler:
        submit(scheduler, first)
        assert entered.wait(timeout=5)
        submit(scheduler, second)
        submit(scheduler, live)
        release.set()
        outputs = [scheduler.outbox.get(timeout=5).data for _ in range(3)]
    assert outputs == ["live", "first", "second"]
    assert first.closed == second.closed == live.closed == [False]
    assert trace == ["first", "second", "live", "first", "first", "second", "second"]


@pytest.mark.parametrize("workers", [1, 2])
def test_abort_waits_for_inflight_work_and_preserves_other_requests(
    workers, monkeypatch
):
    trace = []
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(
        "sglang_omni.scheduling.step_scheduler._ABORTED_REQUEST_ID_LIMIT", 1
    )
    cancelled = Task("cancelled", 3, trace, entered=entered, release=release)
    healthy = Task("healthy", 1, trace)
    with running_scheduler(lambda item: item, max_workers=workers) as scheduler:
        submit(scheduler, cancelled)
        assert entered.wait(timeout=5)
        scheduler.abort("cancelled")
        scheduler.abort("unrelated")
        assert not cancelled.closed
        submit(scheduler, healthy)
        release.set()
        result = scheduler.outbox.get(timeout=5)
        assert result.request_id == "healthy"
    assert trace.count("cancelled") == 1
    assert cancelled.closed == [True]
    assert healthy.closed == [False]


def test_queue_limit_rejects_excess_and_shutdown_closes_accepted_work():
    trace = []
    entered, release = threading.Event(), threading.Event()
    active = Task("active", 3, trace, entered=entered, release=release)
    queued = Task("queued", 1, trace)
    excess = Task("excess", 1, trace)
    with running_scheduler(
        lambda item: item, max_running_requests=1, max_queued_requests=1
    ) as scheduler:
        submit(scheduler, active)
        assert entered.wait(timeout=5)
        submit(scheduler, queued)
        submit(scheduler, excess)
        try:
            rejection = scheduler.outbox.get(timeout=5)
            assert rejection.request_id == "excess"
            assert isinstance(rejection.data, QueueFullError)
            info = scheduler.admin(ADMIN_MODEL_INFO)["data"]
            assert info["active_requests"] == info["queued_requests"] == 1
            assert info["inflight_steps"] == 1
            assert scheduler.admin(ADMIN_PAUSE_GENERATION)["data"]["unsupported"]
            scheduler.stop()
        finally:
            release.set()
    assert active.closed == queued.closed == [True]
    assert excess.closed == [True]
    assert trace == ["active"]


def test_failed_step_releases_capacity_and_next_request_succeeds():
    failed = Task("failed", 1, [], fail=True)
    healthy = Task("healthy", 1, [])
    with running_scheduler(lambda item: item, max_running_requests=1) as scheduler:
        submit(scheduler, failed)
        submit(scheduler, healthy)
        failure = scheduler.outbox.get(timeout=5)
        success = scheduler.outbox.get(timeout=5)
    assert failure.type == "error"
    assert str(failure.data) == "model step failed"
    assert success.type == "result"
    assert success.data == "healthy"
    assert failed.closed == [True]
    assert healthy.closed == [False]


def test_cancelled_queued_ids_are_released_while_work_is_active(monkeypatch):
    class RequestId(str):
        pass

    monkeypatch.setattr(
        "sglang_omni.scheduling.step_scheduler._ABORTED_REQUEST_ID_LIMIT", 1
    )
    entered, release = threading.Event(), threading.Event()
    active = Task("active", 1, [], entered=entered, release=release)
    references = []
    with running_scheduler(
        lambda item: item, max_running_requests=1, max_queued_requests=1
    ) as scheduler:

        def wait_for_queue(count):
            deadline = time.monotonic() + 5
            while scheduler.admin(ADMIN_MODEL_INFO)["data"]["queued_requests"] != count:
                assert time.monotonic() < deadline
                time.sleep(0.001)

        submit(scheduler, active)
        try:
            assert entered.wait(timeout=5)
            for index in range(8):
                queued = Task(RequestId(f"cancelled-{index}"), 1, [])
                references.append(weakref.ref(queued.name))
                submit(scheduler, queued)
                wait_for_queue(1)
                scheduler.abort(queued.name)
                wait_for_queue(0)
                assert queued.closed == [True]
                del queued
            gc.collect()
            assert all(reference() is None for reference in references[:-1])
        finally:
            release.set()


def test_stop_at_idle_boundary_exits_without_another_request(monkeypatch):
    closed = []
    scheduler = StepScheduler(
        lambda item: item, shutdown_callback=lambda: closed.append(True)
    )
    clear = scheduler._wakeup.clear

    def stop_then_clear():
        scheduler.stop()
        clear()

    monkeypatch.setattr(scheduler._wakeup, "clear", stop_then_clear)
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    try:
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert closed == [True]
    finally:
        scheduler._wakeup.set()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_step_scheduler_imports_no_model_runtime():
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
