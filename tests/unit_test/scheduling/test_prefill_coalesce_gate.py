# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the prefill admission-coalescing gate.

The gate holds prefill until ``prefill_coalesce_requests`` are waiting or the
oldest queued request has waited ``prefill_coalesce_wait_ms``. The deadline is
keyed on each request's enqueue time (``_coalesce_enqueue_t``), so partial
upstream admission or an aborted request never restarts the window for the
requests left behind. Chunked prefill in flight and an empty queue pass
straight through. Tested against a stub scheduler with the upstream call
patched to a sentinel.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("sglang")

from sglang_omni.scheduling import omni_scheduler  # noqa: E402
from sglang_omni.scheduling.omni_scheduler import OmniScheduler  # noqa: E402

_UPSTREAM_BATCH = object()


def _req(enqueue_t: float | None):
    if enqueue_t is None:
        return SimpleNamespace()
    return SimpleNamespace(_coalesce_enqueue_t=enqueue_t)


class _StubScheduler:
    """The attribute surface get_new_batch_prefill touches."""

    def __init__(self, *, coalesce_requests: int, wait_ms: float = 60.0) -> None:
        self.prefill_coalesce_requests = coalesce_requests
        self.prefill_coalesce_wait_s = wait_ms / 1e3
        self.chunked_req = None
        self.waiting_queue: list = []
        self.running_batch = SimpleNamespace(is_empty=lambda: False)

    def get_new_batch_prefill(self):
        return OmniScheduler.get_new_batch_prefill(self)


@pytest.fixture()
def upstream():
    with mock.patch.object(
        omni_scheduler._Upstream,
        "get_new_batch_prefill",
        return_value=_UPSTREAM_BATCH,
    ) as patched:
        yield patched


@pytest.fixture()
def clock():
    with mock.patch.object(omni_scheduler.time, "perf_counter") as patched:
        patched.return_value = 100.0
        yield patched


def test_disabled_gate_passes_through(upstream):
    sched = _StubScheduler(coalesce_requests=0)
    sched.waiting_queue = [_req(0.0)]
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_chunked_prefill_bypasses_gate(upstream):
    sched = _StubScheduler(coalesce_requests=8)
    sched.waiting_queue = [_req(0.0)]
    sched.chunked_req = object()
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_empty_queue_passes_through(upstream):
    sched = _StubScheduler(coalesce_requests=8)
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_full_batch_passes_through_immediately(upstream):
    sched = _StubScheduler(coalesce_requests=4)
    sched.waiting_queue = [_req(100.0)] * 4
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_small_queue_is_held_until_oldest_expires(upstream, clock):
    sched = _StubScheduler(coalesce_requests=8, wait_ms=60.0)
    sched.waiting_queue = [_req(100.0), _req(100.01)]

    clock.return_value = 100.03  # oldest has waited 30ms of the 60ms window
    assert sched.get_new_batch_prefill() is None

    clock.return_value = 100.07  # oldest past the deadline
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH
    upstream.assert_called_once()


def test_reaching_target_releases_before_deadline(upstream, clock):
    sched = _StubScheduler(coalesce_requests=3, wait_ms=60.0)
    sched.waiting_queue = [_req(100.0)]
    assert sched.get_new_batch_prefill() is None

    sched.waiting_queue = [_req(100.0)] * 3  # target reached, clock unchanged
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_partial_admission_leftovers_keep_their_deadline(upstream, clock):
    # Note: (maydomine) leftovers of a partially admitted wave keep their old
    # stamps and must not re-wait a fresh window.
    sched = _StubScheduler(coalesce_requests=8, wait_ms=60.0)
    clock.return_value = 100.1
    sched.waiting_queue = [_req(100.0), _req(100.02)]  # both past the deadline
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_abort_does_not_hand_newcomers_an_expired_deadline(upstream, clock):
    # Note: (maydomine) a fresh arrival after an abort waits its own window
    # rather than inheriting the nearly expired one.
    sched = _StubScheduler(coalesce_requests=8, wait_ms=60.0)
    clock.return_value = 100.1
    sched.waiting_queue = [_req(100.09)]  # newcomer, 10ms old
    assert sched.get_new_batch_prefill() is None


def test_idle_loop_bypasses_gate(upstream):
    # Note: (maydomine) no decode in flight: holding amortizes nothing, only
    # costs TTFB.
    sched = _StubScheduler(coalesce_requests=8)
    sched.waiting_queue = [_req(0.0)]
    sched.running_batch = None
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH

    sched.running_batch = SimpleNamespace(is_empty=lambda: True)
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_real_partial_admission_cycle_releases_leftovers_immediately(clock):
    # Note: (Jiaxin Deng) real admit cycle: upstream pops only the head of an
    # expired wave; the leftover keeps its stamp and releases on the next pass.
    def _admit_head(self):
        self.waiting_queue.pop(0)
        return _UPSTREAM_BATCH

    sched = _StubScheduler(coalesce_requests=8, wait_ms=60.0)
    clock.return_value = 100.07
    leftover = _req(100.005)
    sched.waiting_queue = [_req(100.0), leftover]
    with mock.patch.object(
        omni_scheduler._Upstream,
        "get_new_batch_prefill",
        autospec=True,
        side_effect=_admit_head,
    ) as patched:
        assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH
        assert sched.waiting_queue == [leftover]
        assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH
        assert patched.call_count == 2
        assert sched.waiting_queue == []


def test_newcomer_after_queue_drain_waits_its_own_window(upstream, clock):
    # Note: (Jiaxin Deng) once the arming request leaves the queue (abort's
    # queue filtering), a stamp-on-miss newcomer ages from its own arrival.
    sched = _StubScheduler(coalesce_requests=8, wait_ms=60.0)
    clock.return_value = 100.0
    armer = _req(None)
    sched.waiting_queue = [armer]
    assert sched.get_new_batch_prefill() is None
    assert armer._coalesce_enqueue_t == 100.0

    clock.return_value = 100.059  # abort just before the armer's deadline
    sched.waiting_queue.remove(armer)
    newcomer = _req(None)
    sched.waiting_queue.append(newcomer)
    assert sched.get_new_batch_prefill() is None
    assert newcomer._coalesce_enqueue_t == 100.059

    clock.return_value = 100.07  # past the armer's window, inside the newcomer's
    assert sched.get_new_batch_prefill() is None

    clock.return_value = 100.12  # newcomer's own deadline expires
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_unstamped_request_is_stamped_and_eventually_released(upstream, clock):
    # Note: (maydomine) stamp-on-miss: held at first observation, released
    # within the wait deadline (never a K-only unbounded hold).
    sched = _StubScheduler(coalesce_requests=8, wait_ms=60.0)
    clock.return_value = 200.0
    req = _req(None)
    sched.waiting_queue = [req]
    assert sched.get_new_batch_prefill() is None
    assert req._coalesce_enqueue_t == 200.0

    clock.return_value = 200.03  # inside the window
    assert sched.get_new_batch_prefill() is None

    clock.return_value = 200.07  # past the deadline
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH
