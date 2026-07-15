# SPDX-License-Identifier: Apache-2.0
"""Contract tests for PreparedRequestQueue. CPU-only."""

from sglang_omni.scheduling.prepared_request_queue import PreparedRequestQueue


def _active_queue() -> PreparedRequestQueue:
    q: PreparedRequestQueue = PreparedRequestQueue()
    q.set_context(object())
    return q


def test_begin_then_publish_stores_and_pop_returns():
    q = _active_queue()
    assert q.begin("a") is not None
    assert "a" in q.snapshot().inflight
    assert q.publish("a", "PREP-a") is True
    assert "a" not in q.snapshot().inflight
    assert q.pop("a") == "PREP-a"
    assert q.pop("a") is None


def test_abort_while_inflight_then_publish_drops():
    q = _active_queue()
    q.begin("a")
    q.abort("a")
    assert "a" in q.snapshot().aborted
    assert q.publish("a", "PREP-a") is False
    assert "a" not in q.snapshot().prepared
    assert "a" not in q.snapshot().aborted


def test_abort_published_drops_it():
    q = _active_queue()
    q.begin("a")
    q.publish("a", "PREP-a")
    assert "a" in q.snapshot().prepared
    q.abort("a")
    assert "a" not in q.snapshot().prepared
    assert "a" not in q.snapshot().aborted


def test_abort_unknown_request_is_noop():
    q = _active_queue()
    q.abort("ghost")
    assert not q.snapshot().aborted
    assert not q.snapshot().inflight
    assert not q.snapshot().prepared


def test_fail_inflight_discards_and_stores_nothing():
    q = _active_queue()
    q.begin("a")
    q.fail_inflight("a")
    assert not q.snapshot().inflight
    assert not q.snapshot().aborted
    assert "a" not in q.snapshot().prepared


def test_set_and_clear_context_reset_state():
    q = _active_queue()
    q.begin("a")
    q.publish("a", "PREP-a")
    q.begin("b")
    q.set_context(object())
    assert (
        not q.snapshot().prepared
        and not q.snapshot().inflight
        and not q.snapshot().aborted
    )
    q.begin("c")
    q.clear_context()
    assert q.snapshot().context is None
    assert (
        not q.snapshot().prepared
        and not q.snapshot().inflight
        and not q.snapshot().aborted
    )


def test_begin_without_context_returns_none_and_no_inflight():
    q: PreparedRequestQueue = PreparedRequestQueue()
    assert q.begin("a") is None
    assert "a" not in q.snapshot().inflight


def test_publish_without_begin_drops():
    q = _active_queue()
    assert q.publish("a", "PREP-a") is False
    assert "a" not in q.snapshot().prepared


def test_publish_after_reset_drops():
    q = _active_queue()
    q.begin("a")
    q.set_context(object())
    assert q.publish("a", "PREP-a") is False
    assert "a" not in q.snapshot().prepared
