# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for collective ordering during cancellation and shutdown."""

import threading

import pytest

from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.tensor_parallel_scheduler import TensorParallelScheduler


def request(request_id):
    return IncomingMessage(request_id=request_id, type="new_request", data=request_id)


def test_asymmetric_abort_preserves_collective_order():
    collective = threading.Barrier(2, timeout=3)
    entered = threading.Event()
    calls = [[], []]

    def compute(rank, request_id):
        calls[rank].append(request_id)
        entered.set()
        collective.wait()
        return request_id

    schedulers = [
        TensorParallelScheduler(
            lambda value, rank=rank: compute(rank, value),
            shutdown_join_timeout_s=5.0,
        )
        for rank in range(2)
    ]
    threads = [
        threading.Thread(target=scheduler.start, daemon=True)
        for scheduler in schedulers
    ]
    for scheduler in schedulers:
        scheduler.inbox.put(request("cancelled"))
        scheduler.inbox.put(request("next"))
    try:
        threads[0].start()
        assert entered.wait(2)
        # Rank zero is inside its first collective while rank one has not started. Rank one must not skip the cancelled queued forward.
        schedulers[0].abort("cancelled")
        schedulers[1].abort("cancelled")
        # Shutdown must also drain the next dispatched request on both ranks, including a follower whose compute thread has not started yet.
        for scheduler in schedulers:
            scheduler.stop()
        threads[1].start()
        for scheduler in schedulers:
            assert scheduler.outbox.get(timeout=4).data == "cancelled"
            assert scheduler.outbox.get(timeout=4).data == "next"
        assert calls == [["cancelled", "next"], ["cancelled", "next"]]
    finally:
        for scheduler in schedulers:
            scheduler.stop()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=4)
            else:
                pass
    assert all(not thread.is_alive() for thread in threads)


def test_shutdown_drains_work_before_cleanup_on_compute_thread():
    entered = threading.Event()
    release = threading.Event()
    events = []
    cleanup_threads = []

    def compute(request_id):
        if request_id == "active":
            entered.set()
            assert release.wait(3)
        else:
            pass
        events.append(request_id)
        return request_id

    def cleanup():
        events.append("cleanup")
        cleanup_threads.append(threading.get_ident())

    scheduler = TensorParallelScheduler(
        compute, shutdown_join_timeout_s=5.0, shutdown_callback=cleanup
    )
    thread = threading.Thread(target=scheduler.start, daemon=True)
    scheduler.inbox.put(request("active"))
    scheduler.inbox.put(request("queued"))
    thread.start()
    try:
        assert entered.wait(2)
        scheduler.stop()
        scheduler.stop()
        assert events == []
    finally:
        release.set()
        scheduler.stop()
        thread.join(timeout=4)
    assert not thread.is_alive()
    assert events == ["active", "queued", "cleanup"]
    assert cleanup_threads == [thread.ident]
    scheduler.stop()
    assert events.count("cleanup") == 1


def test_stop_before_thread_start_still_drains_and_cleans_up():
    events = []
    scheduler = TensorParallelScheduler(
        lambda value: events.append(value),
        shutdown_join_timeout_s=5.0,
        shutdown_callback=lambda: events.append("cleanup"),
    )
    scheduler.abort("queued")
    scheduler.inbox.put(request("queued"))
    scheduler.stop()
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert events == ["queued", "cleanup"]


def test_failure_is_reported_while_cleanup_is_blocked():
    cleanup_entered = threading.Event()
    release_cleanup = threading.Event()
    failures = []
    errors = []

    def compute(value):
        raise RuntimeError("rank-local allocation failed")

    def cleanup():
        cleanup_entered.set()
        assert release_cleanup.wait(3)

    scheduler = TensorParallelScheduler(
        compute, shutdown_join_timeout_s=5.0, shutdown_callback=cleanup
    )
    scheduler.failure_callback = failures.append
    scheduler.inbox.put(request("failed"))

    def run():
        try:
            scheduler.start()
        except RuntimeError as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert cleanup_entered.wait(2)
        assert thread.is_alive()
        assert len(failures) == 1
        assert str(failures[0].__cause__) == "rank-local allocation failed"
        assert scheduler.shutdown_join_timeout_s is None
        assert scheduler.outbox.get_nowait().type == "error"
    finally:
        release_cleanup.set()
        thread.join(timeout=4)
    assert not thread.is_alive()
    assert errors == failures


def test_forward_failure_does_not_reuse_collectives_for_the_next_request():
    calls = []

    def compute(request_id):
        calls.append(request_id)
        raise RuntimeError("rank-local allocation failed")

    scheduler = TensorParallelScheduler(
        compute,
        shutdown_join_timeout_s=5.0,
        shutdown_callback=lambda: calls.append("cleanup"),
    )
    scheduler.inbox.put(request("failed"))
    scheduler.inbox.put(request("next"))
    with pytest.raises(RuntimeError, match="collective sequence cannot be reused"):
        scheduler.start()
    assert calls == ["failed", "cleanup"]
    error = scheduler.outbox.get_nowait()
    assert error.request_id == "failed"
    assert error.type == "error"
