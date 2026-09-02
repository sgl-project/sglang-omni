# SPDX-License-Identifier: Apache-2.0
"""CI invariants over the allocator workload harness.

Fast, seeded, assertion-bearing versions of the fragmentation experiments in
benchmarks/comm/allocator_workloads.py. The parameter sweeps that feed the
RFC #287 review response live in that module's CLI; these tests pin the
structural facts the sweeps rely on.
"""

from __future__ import annotations

import asyncio

import pytest

from benchmarks.comm.allocator_workloads import (
    ALLOCATORS,
    BuddyAllocator,
    FirstFitAllocator,
    SizeClassAllocator,
    adversarial_schedule,
    bimodal_schedule,
    fixed_schedule,
    run_schedule,
)
from sglang_omni.relay.cuda_ipc import _ContiguousSlotAllocator


@pytest.mark.comm_invariant
def test_fixed_schedule_conserves_slots() -> None:
    allocator = FirstFitAllocator(slot_count=64, slot_size=2**16)
    schedule = fixed_schedule(count=60, size_bytes=3 * 2**16, hold_ticks=4)
    result = asyncio.run(run_schedule(allocator, schedule))
    assert result.unsatisfiable == []
    assert len(result.requests) == 60
    assert all(r.grant_tick is not None for r in result.requests)
    # Everything released: a leak or double-release would break this (release
    # itself raises on double-free).
    assert allocator.layout_snapshot()["free_slots"] == 64
    assert allocator.layout_snapshot()["largest_free_run"] == 64


@pytest.mark.comm_invariant
def test_bimodal_schedule_all_requests_eventually_granted() -> None:
    allocator = FirstFitAllocator(slot_count=256, slot_size=2**16)
    schedule = bimodal_schedule(count=200, seed=7)
    result = asyncio.run(run_schedule(allocator, schedule))
    summary = result.summary()
    assert summary["unsatisfiable"] == 0
    assert summary["granted"] == 200
    # Structural bound, not wall-clock: regression guard at ~2x the observed
    # p99 (119 virtual ticks for this seed/pool at first commit).
    assert summary["wait_ticks"]["p99"] <= 250


@pytest.mark.comm_invariant
def test_adversarial_schedule_provokes_contiguity_wait() -> None:
    """The RFC #287 debate, as a regression test: a comb layout makes free
    capacity plentiful while no contiguous run fits, and the allocator's
    layout metrics classify that wait as fragmentation-induced."""
    allocator = FirstFitAllocator(slot_count=64, slot_size=2**16)
    schedule = adversarial_schedule(
        slot_count=64, slot_size=2**16, large_run=4, large_count=4
    )
    result = asyncio.run(run_schedule(allocator, schedule))
    summary = result.summary()
    contiguity = [r for r in result.requests if r.contiguity_wait]
    assert summary["unsatisfiable"] == 0
    assert contiguity, "comb workload must provoke a contiguity-induced wait"
    for request in contiguity:
        assert request.last_failed_free_slots >= request.num_slots
        assert request.last_failed_largest_free_run < request.num_slots


@pytest.mark.parametrize("allocator_name", ["buddy", "size_class"])
@pytest.mark.comm_invariant
def test_alternative_allocators_conserve_slots(allocator_name: str) -> None:
    allocator = ALLOCATORS[allocator_name](slot_count=64, slot_size=2**16)
    schedule = fixed_schedule(count=40, size_bytes=2**16, hold_ticks=3)
    result = asyncio.run(run_schedule(allocator, schedule))
    assert result.unsatisfiable == []
    assert allocator.layout_snapshot()["free_slots"] == 64


@pytest.mark.comm_invariant
def test_buddy_rounds_up_and_merges() -> None:
    async def _run() -> None:
        allocator = BuddyAllocator(slot_count=8, slot_size=64)
        three = await allocator.acquire_async(3)  # rounds to 4 slots
        assert allocator.layout_snapshot()["free_slots"] == 4
        allocator.release(three.offset, 3)
        assert allocator.layout_snapshot()["free_slots"] == 8
        assert allocator.layout_snapshot()["largest_free_run"] == 8

    asyncio.run(_run())


@pytest.mark.comm_invariant
def test_size_class_isolation_blocks_only_hot_class() -> None:
    async def _run() -> None:
        allocator = SizeClassAllocator(
            slot_count=16,
            slot_size=64,
            class_fractions={1: 0.25, 2: 0.25},
        )
        # Exhaust the 1-slot class (4 blocks).
        singles = [await allocator.acquire_async(1) for _ in range(4)]
        waiter = asyncio.create_task(allocator.acquire_async(1))
        for _ in range(10):
            await asyncio.sleep(0)
        assert not waiter.done(), "hot class exhausted -> capacity wait"
        # Other classes stay available while the 1-slot class is saturated.
        pair = await allocator.acquire_async(2)
        allocator.release(singles[0].offset, 1)
        granted = await asyncio.wait_for(waiter, timeout=1.0)
        assert granted.wait_rounds >= 1
        allocator.release(pair.offset, 2)

    asyncio.run(_run())


@pytest.mark.comm_invariant
def test_acquire_wakes_only_when_contiguous_run_exists() -> None:
    """Allocator-level comb primitive: free capacity equal to the request is
    not enough; the waiter stays blocked until a contiguous run appears."""

    async def _run() -> None:
        allocator = _ContiguousSlotAllocator(slot_count=4, slot_size=64)
        holds = [await allocator.acquire_async(1) for _ in range(4)]
        # Free slots 0 and 2: two free slots, largest run 1.
        allocator.release(holds[0].offset, 1)
        allocator.release(holds[2].offset, 1)

        waiter = asyncio.create_task(allocator.acquire_async(2, capture_layout=True))
        for _ in range(10):
            await asyncio.sleep(0)
        assert not waiter.done(), "comb must keep a 2-slot request blocked"

        # Freeing slot 1 joins slots 0-2 into a contiguous run and wakes it.
        allocator.release(holds[1].offset, 1)
        allocation = await asyncio.wait_for(waiter, timeout=1.0)
        assert allocation.wait_rounds >= 1
        assert allocation.last_failed_free_slots >= 2
        assert allocation.last_failed_largest_free_run == 1
        allocator.release(allocation.offset, 2)
        allocator.release(holds[3].offset, 1)

    asyncio.run(_run())
