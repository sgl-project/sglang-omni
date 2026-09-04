# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for ThreadedSimpleScheduler abort bookkeeping."""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from typing import Callable, Iterator

import pytest

from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.threaded_simple_scheduler import (
    _ABORTED_REQUEST_ID_LIMIT,
    ThreadedSimpleScheduler,
    _CountingInbox,
)


def _request(
    request_id: str, data=None, message_type: str = "new_request"
) -> IncomingMessage:
    return IncomingMessage(request_id=request_id, type=message_type, data=data)


@contextmanager
def _running(scheduler: ThreadedSimpleScheduler) -> Iterator[None]:
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        yield
    finally:
        scheduler.stop()
        thread.join(timeout=5.0)
        assert not thread.is_alive()


def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("condition was not met before timeout")
        time.sleep(0.01)


@pytest.mark.parametrize("abort_before_enqueue", [True, False])
def test_abort_suppresses_unseen_or_queued_request(
    abort_before_enqueue: bool,
) -> None:
    executed: list[str] = []
    scheduler = ThreadedSimpleScheduler(
        lambda payload: executed.append(payload) or payload, max_concurrency=1
    )

    if abort_before_enqueue:
        scheduler.abort("drop")
    scheduler.inbox.put(_request("drop", "must-not-run"))
    if not abort_before_enqueue:
        scheduler.abort("drop")
    scheduler.inbox.put(_request("live", "must-run"))

    with _running(scheduler):
        result = scheduler.outbox.get(timeout=2.0)
        assert (result.request_id, result.data) == ("live", "must-run")
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.2)

    assert executed == ["must-run"]
    assert not scheduler._has_tombstone("drop")


def test_non_request_message_is_not_tracked_as_queued() -> None:
    scheduler = ThreadedSimpleScheduler(lambda payload: payload, max_concurrency=1)
    scheduler.inbox.put(_request("stream-only", "chunk", "stream_chunk"))

    assert not scheduler.inbox.is_reachable("stream-only")
    scheduler.abort("stream-only")

    assert "stream-only" not in scheduler._queued_aborts
    assert "stream-only" in scheduler._speculative_aborts


def test_abort_cancels_pending_future() -> None:
    scheduler = ThreadedSimpleScheduler(lambda payload: payload, max_concurrency=1)
    future: Future = Future()
    scheduler._pending["running"] = future
    future.add_done_callback(lambda fut: scheduler._finish("running", fut))

    scheduler.abort("running")

    assert future.cancelled()
    assert "running" not in scheduler._pending
    assert future not in scheduler._aborted_futures
    assert not scheduler._has_tombstone("running")


def test_enqueue_promotes_speculative_abort_beyond_cap_reach() -> None:
    executed: list[str] = []
    scheduler = ThreadedSimpleScheduler(
        lambda payload: executed.append(payload) or payload, max_concurrency=1
    )
    scheduler.abort("victim")
    assert "victim" in scheduler._speculative_aborts

    scheduler.enqueue(_request("victim", "must-not-run"))

    assert "victim" in scheduler._queued_aborts
    assert "victim" not in scheduler._speculative_aborts

    for i in range(_ABORTED_REQUEST_ID_LIMIT):
        scheduler.abort(f"newer-{i}")
    assert "victim" in scheduler._queued_aborts

    with _running(scheduler):
        scheduler.inbox.put(_request("live", "must-run"))
        result = scheduler.outbox.get(timeout=5.0)
        assert (result.request_id, result.data) == ("live", "must-run")
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.2)

    assert executed == ["must-run"]
    assert not scheduler._has_tombstone("victim")


def test_enqueue_migration_is_atomic_with_inbox_put() -> None:
    put_registered = threading.Event()
    release_put = threading.Event()
    executed: list[str] = []

    class PausingInbox(_CountingInbox):
        def put(self, *args, **kwargs):
            super().put(*args, **kwargs)
            put_registered.set()
            assert release_put.wait(timeout=5.0)

    scheduler = ThreadedSimpleScheduler(
        lambda payload: executed.append(payload) or payload, max_concurrency=1
    )
    scheduler.inbox = PausingInbox()
    scheduler.abort("victim")

    enqueue_thread = threading.Thread(
        target=scheduler.enqueue,
        args=(_request("victim", "must-not-run"),),
        daemon=True,
    )
    enqueue_thread.start()
    assert put_registered.wait(timeout=5.0)

    acquired = scheduler._lock.acquire(blocking=False)
    try:
        assert not acquired, (
            "enqueue must hold the scheduler lock across inbox.put so cap "
            "reclamation cannot evict a marker before it is promoted"
        )
    finally:
        if acquired:
            scheduler._lock.release()
        release_put.set()
    enqueue_thread.join(timeout=5.0)
    assert not enqueue_thread.is_alive()

    assert "victim" in scheduler._queued_aborts
    for i in range(_ABORTED_REQUEST_ID_LIMIT):
        scheduler.abort(f"newer-{i}")
    assert "victim" in scheduler._queued_aborts

    with _running(scheduler):
        scheduler.inbox.put(_request("live", "must-run"))
        result = scheduler.outbox.get(timeout=5.0)
        assert (result.request_id, result.data) == ("live", "must-run")
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.2)

    assert executed == ["must-run"]
    assert not scheduler._has_tombstone("victim")


def test_queued_aborts_past_cap_are_not_evicted() -> None:
    executed: list[str] = []
    scheduler = ThreadedSimpleScheduler(
        lambda payload: executed.append(payload) or payload, max_concurrency=1
    )
    total = _ABORTED_REQUEST_ID_LIMIT + 100
    for i in range(total):
        request_id = f"req-{i}"
        scheduler.inbox.put(_request(request_id, request_id))
        scheduler.abort(request_id)

    assert len(scheduler._queued_aborts) == total
    assert not scheduler._speculative_aborts

    with _running(scheduler):
        _wait_until(lambda: not scheduler._queued_aborts, timeout=10.0)

    assert executed == []


def test_claimed_abort_survives_speculative_eviction() -> None:
    claimed = threading.Event()
    release_get = threading.Event()
    executed = threading.Event()

    class PausingInbox(_CountingInbox):
        def get(self, *args, **kwargs):
            item = super().get(*args, **kwargs)
            claimed.set()
            assert release_get.wait(timeout=5.0)
            return item

    scheduler = ThreadedSimpleScheduler(
        lambda payload: executed.set() or payload, max_concurrency=1
    )
    scheduler.inbox = PausingInbox()
    scheduler.inbox.put(_request("claimed", "must-not-run"))

    with _running(scheduler):
        try:
            assert claimed.wait(timeout=5.0)
            scheduler.abort("claimed")
            for i in range(_ABORTED_REQUEST_ID_LIMIT):
                scheduler.abort(f"newer-{i}")
            release_get.set()

            _wait_until(lambda: not scheduler.inbox._claimed_counts)
            with pytest.raises(queue.Empty):
                scheduler.outbox.get(timeout=0.3)
            assert not executed.is_set()
            assert not scheduler._has_tombstone("claimed")
        finally:
            release_get.set()


def test_abort_for_reused_id_survives_old_future_completion() -> None:
    started = threading.Event()
    release = threading.Event()
    executed: list[str] = []

    def compute(payload: str) -> str:
        executed.append(payload)
        if payload == "old":
            started.set()
            assert release.wait(timeout=5.0)
        return payload

    scheduler = ThreadedSimpleScheduler(compute, max_concurrency=1)
    with _running(scheduler):
        try:
            scheduler.inbox.put(_request("reused", "old"))
            assert started.wait(timeout=5.0)
            scheduler.abort("reused")
            scheduler.abort("reused")
            assert scheduler._has_tombstone("reused")

            release.set()
            _wait_until(lambda: not scheduler._aborted_futures)
            assert scheduler._has_tombstone("reused")

            scheduler.inbox.put(_request("reused", "new"))
            scheduler.inbox.put(_request("live", "live"))
            assert scheduler.outbox.get(timeout=5.0).data == "live"
            assert executed == ["old", "live"]
            assert not scheduler._has_tombstone("reused")
        finally:
            release.set()


def test_stale_aborted_future_does_not_clear_newer_abort() -> None:
    scheduler = ThreadedSimpleScheduler(lambda payload: payload, max_concurrency=2)
    old_future: Future = Future()
    new_future: Future = Future()
    assert old_future.set_running_or_notify_cancel()
    assert new_future.set_running_or_notify_cancel()

    scheduler._pending["reused"] = old_future
    old_future.add_done_callback(lambda fut: scheduler._finish("reused", fut))
    new_future.add_done_callback(lambda fut: scheduler._finish("reused", fut))

    scheduler.abort("reused")
    scheduler._pending["reused"] = new_future
    scheduler.abort("reused")
    old_future.set_result("old-result")
    new_future.set_result("new-result")

    with pytest.raises(queue.Empty):
        scheduler.outbox.get_nowait()
    assert not scheduler._aborted_futures


def test_stale_finish_keeps_newer_pending_future() -> None:
    scheduler = ThreadedSimpleScheduler(lambda payload: payload, max_concurrency=2)
    old_future: Future = Future()
    new_future: Future = Future()
    scheduler._pending["reused"] = new_future

    old_future.set_result("old-result")
    scheduler._finish("reused", old_future)

    assert scheduler._pending["reused"] is new_future
    assert scheduler.outbox.get(timeout=2.0).data == "old-result"


def test_speculative_aborts_are_fifo_bounded() -> None:
    scheduler = ThreadedSimpleScheduler(lambda payload: payload, max_concurrency=1)
    request_ids = [f"req-{i}" for i in range(_ABORTED_REQUEST_ID_LIMIT)]
    for request_id in request_ids:
        scheduler.abort(request_id)

    scheduler.abort("overflow")

    assert list(scheduler._speculative_aborts) == request_ids[5001:] + ["overflow"]


def test_running_abort_survives_speculative_eviction() -> None:
    started = threading.Event()
    release = threading.Event()

    def compute(payload: str) -> str:
        started.set()
        assert release.wait(timeout=5.0)
        return payload

    scheduler = ThreadedSimpleScheduler(compute, max_concurrency=1)
    with _running(scheduler):
        try:
            scheduler.inbox.put(_request("running", "must-not-return"))
            assert started.wait(timeout=5.0)
            scheduler.abort("running")
            for i in range(_ABORTED_REQUEST_ID_LIMIT + 1):
                scheduler.abort(f"newer-{i}")
            release.set()

            _wait_until(lambda: not scheduler._aborted_futures)
            with pytest.raises(queue.Empty):
                scheduler.outbox.get(timeout=0.2)
        finally:
            release.set()
