# SPDX-License-Identifier: Apache-2.0
"""One lock across the two threads a PD Decode half allocates from.

The allocator's alloc reads the free list, slices it, and writes the
remainder back, with no lock of its own. On a Decode half the scheduler
thread and the comm event loop both call it, and interleaving there hands
the same slots to both.
"""

from __future__ import annotations

import threading

import pytest

from sglang_omni.scheduling.pd_alloc_lock import LockedKVAllocator


class _FakeAllocator:
    """Stands in for the real allocator; only identity matters here."""

    def alloc(self, n):  # pragma: no cover - never called
        return list(range(n))

    def free(self, indices):  # pragma: no cover - never called
        return None


class _RacyAllocator:
    """An allocator with the same read-modify-write shape as upstream's."""

    def __init__(self, size: int) -> None:
        self.free_pages = list(range(size))
        self.page_size = 1
        self.handed_out: list[int] = []

    def alloc(self, need_size: int):
        taken = self.free_pages[:need_size]
        # A thread switch here is what produces the overlap in production.
        threading.current_thread()
        self.free_pages = self.free_pages[need_size:]
        self.handed_out.extend(taken)
        return taken

    def free(self, index) -> None:
        self.free_pages.extend(index)

    def available_size(self) -> int:
        return len(self.free_pages)


def test_two_threads_never_receive_the_same_slot() -> None:
    inner = _RacyAllocator(4096)
    allocator = LockedKVAllocator(inner)
    seen: list[int] = []
    lock = threading.Lock()

    def take() -> None:
        for _ in range(64):
            got = allocator.alloc(4)
            with lock:
                seen.extend(got)

    threads = [threading.Thread(target=take) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(seen) == len(set(seen))


def test_everything_else_is_delegated() -> None:
    inner = _RacyAllocator(16)
    allocator = LockedKVAllocator(inner)

    assert allocator.available_size() == 16
    assert allocator.page_size == 1


def test_a_write_reaches_the_wrapped_allocator() -> None:
    """Upstream code sets attributes on the allocator it was handed."""
    inner = _RacyAllocator(16)
    allocator = LockedKVAllocator(inner)

    allocator.page_size = 4

    assert inner.page_size == 4


def test_freeing_returns_the_slots() -> None:
    inner = _RacyAllocator(8)
    allocator = LockedKVAllocator(inner)

    taken = allocator.alloc(8)
    assert allocator.available_size() == 0

    allocator.free(taken)
    assert allocator.available_size() == 8


def test_every_holder_of_the_allocator_gets_the_same_locked_object() -> None:
    """One lock only helps if no caller keeps an unwrapped alias."""
    from types import SimpleNamespace

    from sglang_omni.scheduling.pd_scheduler import (
        OmniDecodeScheduler,
        _serialize_kv_allocation,
        kv_allocator_holders,
    )

    raw = _FakeAllocator()
    scheduler = OmniDecodeScheduler.__new__(OmniDecodeScheduler)
    scheduler.token_to_kv_pool_allocator = raw
    scheduler.tree_cache = SimpleNamespace(token_to_kv_pool_allocator=raw)

    _serialize_kv_allocation(scheduler)

    holders = kv_allocator_holders(scheduler)
    assert set(holders) == {"scheduler", "tree_cache"}
    assert len({id(a) for a in holders.values()}) == 1
    assert all(isinstance(a, LockedKVAllocator) for a in holders.values())


def test_a_holder_that_cannot_be_rebound_is_an_error() -> None:
    """Skipping it silently would leave an alias outside the lock."""
    from types import SimpleNamespace

    from sglang_omni.scheduling.pd_scheduler import (
        OmniDecodeScheduler,
        _serialize_kv_allocation,
    )

    scheduler = OmniDecodeScheduler.__new__(OmniDecodeScheduler)
    scheduler.token_to_kv_pool_allocator = _FakeAllocator()
    scheduler.tree_cache = SimpleNamespace()  # no allocator attribute

    with pytest.raises(RuntimeError, match="would bypass the lock"):
        _serialize_kv_allocation(scheduler)


def test_a_receiver_built_after_the_wrap_gets_the_locked_allocator() -> None:
    """Ordering matters: a receiver built first would hold the raw object."""
    import queue
    from types import SimpleNamespace

    from sglang_omni.scheduling.pd_scheduler import _serialize_kv_allocation
    from sglang_omni.scheduling.pd_utils import DecodeKVReceiver

    inner = _RacyAllocator(64)
    scheduler = SimpleNamespace(
        token_to_kv_pool_allocator=inner,
        tree_cache=SimpleNamespace(token_to_kv_pool_allocator=inner),
    )
    _serialize_kv_allocation(scheduler)

    receiver = DecodeKVReceiver(
        pool_id="thinker_decode:kv",
        allocator=scheduler.token_to_kv_pool_allocator,
        admissions=queue.SimpleQueue(),
        allowed_resume_schemas=frozenset({"v1"}),
    )

    assert isinstance(receiver._allocator, LockedKVAllocator)


def test_a_lease_release_does_not_touch_the_tree_on_the_calling_thread(
    monkeypatch,
) -> None:
    """release runs on the comm loop; the tree belongs to the scheduler."""
    import queue as _queue

    import sglang.srt.mem_cache.common as common

    from sglang_omni.scheduling.pd_utils import SGLangKVLease

    freed: list[object] = []
    monkeypatch.setattr(
        common, "release_kv_cache", lambda req, cache: freed.append(req)
    )

    req = object()
    due: _queue.SimpleQueue = _queue.SimpleQueue()
    lease = SGLangKVLease(req, object(), due)

    lease.release()

    assert freed == []
    assert due.get_nowait() is req


def test_a_lease_releases_once_however_many_times_it_is_called() -> None:
    import queue as _queue

    from sglang_omni.scheduling.pd_utils import SGLangKVLease

    due: _queue.SimpleQueue = _queue.SimpleQueue()
    lease = SGLangKVLease(object(), object(), due)

    lease.release()
    lease.release()
    lease.release()

    assert due.qsize() == 1


def test_the_scheduler_thread_drains_every_due_release(monkeypatch) -> None:
    import queue as _queue

    from sglang_omni.scheduling import pd_utils

    freed: list[tuple[object, object]] = []
    tree = object()

    import sglang.srt.mem_cache.common as common

    monkeypatch.setattr(
        common, "release_kv_cache", lambda req, cache: freed.append((req, cache))
    )

    due: _queue.SimpleQueue = _queue.SimpleQueue()
    first, second = object(), object()
    due.put(first)
    due.put(second)

    assert pd_utils.drain_due_releases(due, tree) == 2
    assert freed == [(first, tree), (second, tree)]
    assert pd_utils.drain_due_releases(due, tree) == 0


def test_one_failed_release_does_not_strand_the_rest(monkeypatch) -> None:
    """A release that raises must not leave the queue holding the others."""
    import queue as _queue

    import sglang.srt.mem_cache.common as common

    from sglang_omni.scheduling import pd_utils

    freed: list[object] = []

    def flaky(req, cache):
        if req == "bad":
            raise RuntimeError("tree said no")
        freed.append(req)

    monkeypatch.setattr(common, "release_kv_cache", flaky)

    due: _queue.SimpleQueue = _queue.SimpleQueue()
    due.put("bad")
    due.put("good")

    assert pd_utils.drain_due_releases(due, object()) == 2
    assert freed == ["good"]


def _prefill_stage(topology, bindings):
    """A Stage with only what the KV send path reads."""
    from sglang_omni.pipeline.stage.runtime import Stage

    stage = Stage.__new__(Stage)
    stage.name = "thinker_prefill"
    stage._replica_topology = topology
    stage._replica_bindings = bindings
    return stage


def test_an_unreplicated_decode_target_is_sent_to_unchanged() -> None:
    from sglang_omni.pipeline.replicas import ReplicaTopology

    stage = _prefill_stage(ReplicaTopology(replicas={}), {})

    assert stage._resolve_target_instance("req-1", "thinker_decode") == "thinker_decode"


def test_a_replicated_decode_target_follows_the_admission_binding() -> None:
    """The coordinator chose once; the send must not choose again."""
    from sglang_omni.pipeline.replicas import ReplicaTopology

    topology = ReplicaTopology(
        replicas={"thinker_decode": ("thinker_decode@r0", "thinker_decode@r1")}
    )
    stage = _prefill_stage(topology, {"req-1": {"thinker_decode": 1}})

    assert (
        stage._resolve_target_instance("req-1", "thinker_decode") == "thinker_decode@r1"
    )


def test_a_replicated_target_without_a_binding_is_an_error() -> None:
    """Guessing here would send KV where the request is not expected."""
    from sglang_omni.pipeline.replicas import ReplicaTopology

    topology = ReplicaTopology(
        replicas={"thinker_decode": ("thinker_decode@r0", "thinker_decode@r1")}
    )
    stage = _prefill_stage(topology, {})

    with pytest.raises(RuntimeError, match="no replica binding"):
        stage._resolve_target_instance("req-1", "thinker_decode")


def _gated_stage(limit):
    from sglang_omni.pipeline.stage.runtime import Stage

    stage = Stage.__new__(Stage)
    stage._max_inflight_handoffs = limit
    return stage


def test_an_unset_bound_installs_no_gate() -> None:
    """Leaving it unset keeps today's behaviour rather than picking a number."""
    assert _gated_stage(None)._pd_handoff_gate() is None


def test_a_set_bound_installs_a_semaphore_of_that_size() -> None:
    assert _gated_stage(4)._pd_handoff_gate()._value == 4


def test_the_same_gate_is_reused_across_handoffs() -> None:
    """A fresh semaphore per handoff would bound nothing."""
    stage = _gated_stage(4)

    assert stage._pd_handoff_gate() is stage._pd_handoff_gate()


def test_cancelling_while_queued_releases_the_prompt_kv() -> None:
    """The lease exists before the wait, so nothing downstream releases it."""
    import asyncio as _asyncio
    from types import SimpleNamespace

    class _Lease:
        def __init__(self) -> None:
            self.released = 0

        def release(self) -> None:
            self.released += 1

    lease = _Lease()
    transfer = SimpleNamespace(request_id="req-1", lease=lease)
    stage = _gated_stage(1)
    cleared: list[str] = []
    stage._clear_request_state = cleared.append

    async def scenario() -> None:
        gate = stage._pd_handoff_gate()
        await gate.acquire()  # the one permit is taken by another handoff
        queued = _asyncio.create_task(stage._send_kv_transfer(transfer))
        await _asyncio.sleep(0)
        queued.cancel()
        try:
            await queued
        except _asyncio.CancelledError:
            pass

    _asyncio.run(scenario())

    assert lease.released == 1
    assert cleared == ["req-1"]
