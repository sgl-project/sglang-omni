# SPDX-License-Identifier: Apache-2.0
import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


@pytest.mark.parametrize("fails", [False, True])
def test_queue_batch_and_compute_events(monkeypatch, fails):
    def compute(payload):
        if fails:
            raise ValueError("failed")
        return payload

    scheduler = SimpleScheduler(compute)
    clock = iter([1, 3, 8])
    monkeypatch.setattr(
        "sglang_omni.scheduling.simple_scheduler.time.perf_counter",
        lambda: next(clock),
    )
    payload = SimpleNamespace(
        request=SimpleNamespace(metadata={"omni_metrics_enabled": True})
    )
    scheduler.enqueue(IncomingMessage("r", "new_request", payload))
    loop = asyncio.new_event_loop()
    try:
        if fails:
            with pytest.raises(ValueError):
                scheduler.run_single(scheduler.next_message(), loop)
        else:
            scheduler.run_single(scheduler.next_message(), loop)
    finally:
        loop.close()
    events = []
    while not scheduler.outbox.empty():
        event = scheduler.outbox.get_nowait()
        if event.type == "metrics":
            events.append(event.data)
    assert events == [
        {"event": "queue", "duration": 2},
        {"event": "batch", "size": 1},
        {
            "event": "execution",
            "duration": 5,
            "outcome": "error" if fails else "completed",
        },
    ]
    assert not scheduler.metrics_started


def test_concurrent_workers_report_actual_running_requests():
    release = threading.Event()

    def compute(payload):
        assert release.wait(3)
        return payload

    scheduler = SimpleScheduler(compute, max_concurrency=2)
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        for i in range(3):
            scheduler.enqueue(IncomingMessage(str(i), "new_request", i))
        deadline = time.monotonic() + 2
        while (
            scheduler.metrics_snapshot()["num_running_reqs"] != 2
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert scheduler.metrics_snapshot() == {
            "num_running_reqs": 2,
            "num_queue_reqs": 1,
        }
        release.set()
        for _ in range(3):
            scheduler.outbox.get(timeout=3)
        deadline = time.monotonic() + 2
        while (
            scheduler.metrics_snapshot()["num_running_reqs"]
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert scheduler.metrics_snapshot() == {
            "num_running_reqs": 0,
            "num_queue_reqs": 0,
        }
    finally:
        release.set()
        scheduler.stop()
        thread.join(timeout=3)
    assert not thread.is_alive()


def test_waiting_running_and_abort():
    scheduler = None

    def compute(payload):
        assert scheduler.metrics_snapshot() == {
            "num_running_reqs": 1,
            "num_queue_reqs": 1,
        }
        return payload

    scheduler = SimpleScheduler(compute)
    scheduler.enqueue(IncomingMessage("a", "new_request", 1))
    scheduler.enqueue(IncomingMessage("b", "new_request", 2))
    assert scheduler.metrics_snapshot() == {"num_running_reqs": 0, "num_queue_reqs": 2}
    loop = asyncio.new_event_loop()
    try:
        scheduler.run_single(scheduler.next_message(), loop)
    finally:
        loop.close()
    scheduler.abort("b")
    assert scheduler.metrics_snapshot() == {"num_running_reqs": 0, "num_queue_reqs": 0}


def test_batch_failure_clears_running():
    def fail(payloads):
        assert scheduler.metrics_snapshot()["num_running_reqs"] == 2
        raise ValueError("failed")

    scheduler = SimpleScheduler(lambda x: x, batch_compute_fn=fail, max_batch_size=2)
    batch = [IncomingMessage(str(i), "new_request", i) for i in range(2)]
    for message in batch:
        scheduler.enqueue(message)
    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(ValueError):
            scheduler.run_batch(batch, loop)
    finally:
        loop.close()
    assert scheduler.metrics_snapshot() == {"num_running_reqs": 0, "num_queue_reqs": 0}
