# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading

from sglang_omni_v1.scheduling.messages import IncomingMessage
from sglang_omni_v1.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler


def test_threaded_simple_scheduler_runs_requests_concurrently() -> None:
    started: list[str] = []
    lock = threading.Lock()
    both_started = threading.Event()
    release = threading.Event()

    def _compute(payload: str) -> str:
        with lock:
            started.append(payload)
            if len(started) == 2:
                both_started.set()
        assert release.wait(timeout=2.0)
        return payload

    scheduler = ThreadedSimpleScheduler(_compute, max_concurrency=2)
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        scheduler.inbox.put(IncomingMessage("req-1", "new_request", "one"))
        scheduler.inbox.put(IncomingMessage("req-2", "new_request", "two"))

        assert both_started.wait(timeout=2.0)
        release.set()

        results = {
            scheduler.outbox.get(timeout=2.0).request_id,
            scheduler.outbox.get(timeout=2.0).request_id,
        }
        assert results == {"req-1", "req-2"}
    finally:
        release.set()
        scheduler.stop()
        thread.join(timeout=2.0)


def test_threaded_simple_scheduler_reports_errors() -> None:
    def _compute(payload: str) -> str:
        raise RuntimeError(payload)

    scheduler = ThreadedSimpleScheduler(_compute, max_concurrency=1)
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        scheduler.inbox.put(IncomingMessage("req-err", "new_request", "boom"))
        message = scheduler.outbox.get(timeout=2.0)
        assert message.request_id == "req-err"
        assert message.type == "error"
        assert isinstance(message.data, RuntimeError)
    finally:
        scheduler.stop()
        thread.join(timeout=2.0)
