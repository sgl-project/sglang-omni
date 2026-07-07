# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the T7 PreparedRequestQueue (RFC #661, §13). CPU-only.

The seven §13 scenarios plus a thread-storm: the queue exists only to be safe
across the asyncio-preprocessing stage, the AR-engine thread, and the abort
path, so a single concurrency test guards the property the others cannot.
"""

from __future__ import annotations

import threading

from sglang_omni.scheduling.prepared_request_queue import PreparedRequestQueue


def _active() -> PreparedRequestQueue:
    q: PreparedRequestQueue = PreparedRequestQueue()
    q.set_context(object())
    return q


def test_abort_while_inflight_makes_publish_drop() -> None:
    q = _active()
    q.begin("a")
    q.abort("a")
    assert "a" in q.aborted
    assert q.publish("a", "PREP-a") is False
    assert "a" not in q.prepared
    assert "a" not in q.aborted


def test_abort_after_publish_drops_the_prepared_entry() -> None:
    q = _active()
    q.begin("a")
    assert q.publish("a", "PREP-a") is True
    q.abort("a")
    assert "a" not in q.prepared
    assert "a" not in q.aborted


def test_publish_then_pop_returns_the_payload() -> None:
    q = _active()
    q.begin("a")
    q.publish("a", "PREP-a")
    assert "a" not in q.inflight
    assert q.pop("a") == "PREP-a"


def test_abort_for_idle_id_is_a_noop() -> None:
    q = _active()
    q.abort("ghost")
    assert not q.aborted
    assert not q.inflight
    assert not q.prepared


def test_fail_inflight_leaves_nothing_behind() -> None:
    q = _active()
    q.begin("a")
    q.fail_inflight("a")
    assert not q.inflight
    assert not q.aborted
    assert "a" not in q.prepared


def test_begin_clear_context_strands_no_stale_inflight() -> None:
    q = _active()
    q.begin("a")
    q.publish("a", "PREP-a")
    q.begin("b")
    q.clear_context()
    assert q.context is None
    assert not q.prepared and not q.inflight and not q.aborted
    # note (Xinhao Tan): with no context, begin is a no-op so no in-flight id is added
    assert q.begin("c") is None
    assert "c" not in q.inflight


def test_pop_absent_id_returns_none() -> None:
    q = _active()
    assert q.pop("missing") is None


def test_concurrent_handoffs_do_not_lose_or_cross_payloads() -> None:
    q: PreparedRequestQueue = PreparedRequestQueue()
    q.set_context(object())
    n = 200
    taken: dict[int, str] = {}
    taken_lock = threading.Lock()
    start = threading.Barrier(2)

    def producer() -> None:
        start.wait()
        for i in range(n):
            rid = str(i)
            q.begin(rid)
            q.publish(rid, f"PREP-{i}")

    def consumer() -> None:
        start.wait()
        remaining = set(range(n))
        while remaining:
            for i in list(remaining):
                prepared = q.pop(str(i))
                if prepared is not None:
                    with taken_lock:
                        taken[i] = prepared
                    remaining.discard(i)

    threads = [threading.Thread(target=producer), threading.Thread(target=consumer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert taken == {i: f"PREP-{i}" for i in range(n)}
