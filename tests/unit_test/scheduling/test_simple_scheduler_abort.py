# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for SimpleScheduler dropping aborted requests before batch compute."""

from __future__ import annotations

import queue
import threading
from contextlib import contextmanager
from typing import Iterator

from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def request(request_id: str) -> IncomingMessage:
    return IncomingMessage(request_id=request_id, type="new_request", data=request_id)


@contextmanager
def running(scheduler: SimpleScheduler) -> Iterator[None]:
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        yield
    finally:
        scheduler.stop()
        thread.join(timeout=5.0)
        assert not thread.is_alive()


def drain(outbox: queue.Queue[OutgoingMessage], count: int) -> list[OutgoingMessage]:
    outputs = [outbox.get(timeout=2.0) for _ in range(count)]
    try:
        outputs.append(outbox.get(timeout=0.2))
    except queue.Empty:
        pass
    return outputs


def test_requests_aborted_before_compute_skip_the_batch_and_free_their_slots() -> None:
    batches: list[list[str]] = []

    def batch_compute(payloads: list[str]) -> list[str]:
        batches.append(list(payloads))
        return [payload.upper() for payload in payloads]

    scheduler = SimpleScheduler(
        lambda payload: payload,
        batch_compute_fn=batch_compute,
        max_batch_size=2,
        max_batch_wait_ms=50,
    )
    for request_id in ("a", "b", "c", "d"):
        scheduler.enqueue(request(request_id))
    scheduler.abort("b")
    scheduler.abort("c")

    with running(scheduler):
        outputs = drain(scheduler.outbox, count=2)

    assert batches == [["a", "d"]]
    assert [(out.request_id, out.data) for out in outputs] == [("a", "A"), ("d", "D")]
    assert scheduler.aborted == set()


def test_abort_during_the_batch_wait_window_drops_the_request() -> None:
    scheduler = SimpleScheduler(
        lambda payload: payload,
        batch_compute_fn=lambda payloads: list(payloads),
        max_batch_size=4,
        max_batch_wait_ms=10,
    )
    scheduler.inbox.put(request("b"))
    scheduler.abort("a")

    batch = scheduler.collect_batch(request("a"))

    assert [msg.request_id for msg in batch] == ["b"]
    assert scheduler.aborted == set()


def test_abort_during_compute_still_suppresses_the_result() -> None:
    scheduler = SimpleScheduler(
        lambda payload: payload, max_batch_size=2, max_batch_wait_ms=50
    )

    def batch_compute(payloads: list[str]) -> list[str]:
        scheduler.abort("b")
        return [payload.upper() for payload in payloads]

    scheduler.batch_fn = batch_compute
    scheduler.enqueue(request("a"))
    scheduler.enqueue(request("b"))

    with running(scheduler):
        outputs = drain(scheduler.outbox, count=1)

    assert [(out.request_id, out.data) for out in outputs] == [("a", "A")]
    assert scheduler.aborted == set()
