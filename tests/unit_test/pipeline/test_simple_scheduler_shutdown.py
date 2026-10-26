# SPDX-License-Identifier: Apache-2.0
"""Model cleanup cannot race an active scheduler forward."""

import threading

import pytest

from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


@pytest.mark.parametrize("concurrency", [1, 2])
def test_cleanup_waits_for_active_compute(concurrency):
    entered = threading.Event()
    release = threading.Event()
    events = []

    def compute(value):
        entered.set()
        assert release.wait(3)
        events.append(value)

    scheduler = SimpleScheduler(
        compute,
        max_concurrency=concurrency,
        shutdown_callback=lambda: events.append("cleanup"),
        shutdown_on_compute_thread=True,
    )
    scheduler.inbox.put(IncomingMessage("r", "new_request", "forward"))
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        assert entered.wait(2)
        scheduler.stop()
        assert events == []
    finally:
        release.set()
        scheduler.stop()
        thread.join(timeout=4)
    assert not thread.is_alive()
    assert events == ["forward", "cleanup"]
    scheduler.stop()
    assert events.count("cleanup") == 1


def test_stop_before_start_cleans_up_once_without_running_closed_model():
    events = []
    scheduler = SimpleScheduler(
        lambda value: events.append(value),
        shutdown_callback=lambda: events.append("cleanup"),
        shutdown_on_compute_thread=True,
    )
    scheduler.inbox.put(IncomingMessage("r", "new_request", "forward"))
    scheduler.stop()
    scheduler.start()
    scheduler.stop()
    assert events == ["cleanup"]
