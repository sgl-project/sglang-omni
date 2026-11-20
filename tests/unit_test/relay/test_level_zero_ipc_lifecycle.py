# SPDX-License-Identifier: Apache-2.0
"""What the Level-Zero IPC relay keeps or gives back when a transfer goes wrong."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import sglang_omni.relay.level_zero_ipc as level_zero_ipc
import sglang_omni.relay.level_zero_ipc_operations as level_zero_ipc_operations
import sglang_omni.relay.level_zero_ipc_peers as level_zero_ipc_peers
from sglang_omni.comm.engine import CommEngine
from sglang_omni.comm.router import CommRouter
from sglang_omni.platforms.level_zero import AddressRange, ExportedSegment
from sglang_omni.relay.level_zero_ipc import LevelZeroIpcRelay
from sglang_omni.relay.level_zero_ipc_metadata import (
    PeerPoolReference,
    TransferPlacement,
)
from sglang_omni.relay.level_zero_ipc_operations import (
    LevelZeroIpcGetOperation,
    LevelZeroIpcPutOperation,
)
from sglang_omni.relay.level_zero_ipc_peers import (
    MAX_MAPPED_PEER_POOLS,
    PeerPool,
    PeerPoolMappings,
)
from sglang_omni.relay.level_zero_rendezvous import HandleRendezvous
from sglang_omni.relay.slot_pool import ContiguousSlotAllocator

SLOT_SIZE_KB = 64
POOL_SIZE_MB = 8
POOL_BYTES = 262144
# Long enough that a stuck mapping fails the test rather than hanging it.
PEER_TIMEOUT_S = 60


def peer_reference(*, segment_offset_bytes: int, pool_bytes: int) -> PeerPoolReference:
    return PeerPoolReference(
        pool_id="sender:1:abc",
        socket_name="sglang-omni-level-zero-1-abc",
        segment_offset_bytes=segment_offset_bytes,
        pool_bytes=pool_bytes,
        source_device_index=0,
    )


def test_a_bad_wait_thread_setting_fails_before_any_context_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts: list[str] = []
    monkeypatch.setenv("SGLANG_OMNI_LEVEL_ZERO_IPC_WAIT_THREADS", "0")
    monkeypatch.setattr(
        level_zero_ipc, "LevelZeroContext", lambda: contexts.append("created")
    )

    with pytest.raises(ValueError, match="must be positive"):
        LevelZeroIpcRelay(
            engine_id="sender",
            device="xpu:0",
            slot_size_kb=SLOT_SIZE_KB,
            pool_size_mb=POOL_SIZE_MB,
        )
    assert contexts == []


def test_a_cancelled_ack_wait_fails_the_relay_and_keeps_the_slots() -> None:
    """The receiver may still be reading, and cancelling the wait also drops the
    ACK that would release the slots, so the relay must stop handing them out."""

    async def run() -> None:
        allocator = ContiguousSlotAllocator(slot_count=2, slot_size=8)
        allocation = await allocator.acquire_async(1)
        failures: list[BaseException] = []
        operation = LevelZeroIpcPutOperation(
            {},
            on_failure=failures.append,
            allocator=allocator,
            placement=TransferPlacement(
                size_bytes=8,
                offset_bytes=allocation.offset,
                slot_index=0,
                slot_size_bytes=8,
                num_slots=1,
            ),
            request_id="r",
        )
        waiter = asyncio.create_task(operation.wait_for_completion(timeout=30))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert len(failures) == 1
        assert allocator.free_slots == 1

    asyncio.run(run())


def test_cancelling_the_engine_ack_wait_fails_the_relay_and_keeps_the_slots() -> None:
    """The engine cancels its own ACK wait and only reports it through
    record_sender_cancellation, so the operation's wait never runs; the relay must
    still fail rather than let later sends wait on slots that never come back."""

    async def run() -> None:
        engine = CommEngine(
            CommRouter(
                stage_name="sender",
                gpu_id=None,
                same_process_targets=set(),
                gpu_stage_names=set(),
            )
        )
        allocator = ContiguousSlotAllocator(slot_count=2, slot_size=8)
        allocation = await allocator.acquire_async(1)
        failures: list[BaseException] = []
        operation = LevelZeroIpcPutOperation(
            {},
            on_failure=failures.append,
            allocator=allocator,
            placement=TransferPlacement(
                size_bytes=8,
                offset_bytes=allocation.offset,
                slot_index=0,
                slot_size_bytes=8,
                num_slots=1,
            ),
            request_id="transfer",
        )
        engine.register_pending("transfer", [operation])
        pending_task = engine.arm_pending("transfer")
        await asyncio.sleep(0)
        assert not engine.pending["transfer"].ack.done()

        pending_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending_task
        assert [type(error) for error in failures] == [asyncio.CancelledError]
        assert allocator.free_slots == 1
        assert "transfer" not in engine.pending

        # A second report, or a late wait, neither fails the relay again nor frees.
        operation.record_sender_cancellation(asyncio.CancelledError())
        await operation.wait_for_completion(timeout=1)
        assert len(failures) == 1
        assert allocator.free_slots == 1

    asyncio.run(run())


def test_slots_granted_to_a_cancelled_sender_go_back_to_the_pool() -> None:
    async def run() -> None:
        allocator = ContiguousSlotAllocator(slot_count=1, slot_size=8)
        held = await allocator.acquire_async(1)
        relay = LevelZeroIpcRelay.__new__(LevelZeroIpcRelay)
        relay.failed_error = None
        relay.failed_event = asyncio.Event()
        sender = asyncio.create_task(relay.acquire_slots(allocator, 1))
        await asyncio.sleep(0)
        # The grant lands on the loop's next turn, after the sender is cancelled.
        allocator.release(held.offset, 1)
        sender.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sender
        await asyncio.sleep(0)
        assert allocator.free_slots == 1

    asyncio.run(run())


def test_close_waits_for_a_mapping_still_being_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled read leaves its mapping worker running, and the worker uses the
    context that close destroys, so close has to drain it and unmap what it made."""
    events: list[str] = []

    def map_slowly() -> PeerPool:
        time.sleep(0.2)
        events.append("mapped")
        return PeerPool(
            peer_pointer=7, pool_tensor=torch.empty(0), socket_name="sender-socket"
        )

    relay = LevelZeroIpcRelay.__new__(LevelZeroIpcRelay)
    relay.engine_id = "receiver"
    relay.device_index = 0
    relay.rendezvous = None
    relay.exported_segment = None
    relay.pool_tensor = None
    relay.pool_pointer = None
    relay.level_zero = SimpleNamespace(
        close_peer_segment=lambda pointer: events.append(f"unmapped {pointer}"),
        destroy=lambda: events.append("destroyed"),
    )
    relay.wait_executor = ThreadPoolExecutor(max_workers=1)
    relay.peer_pools = PeerPoolMappings(
        level_zero=relay.level_zero,
        wait_executor=relay.wait_executor,
        engine_id=relay.engine_id,
    )
    relay.peer_pools.mappings["pool"] = relay.wait_executor.submit(map_slowly)
    monkeypatch.setattr(torch.xpu, "synchronize", lambda device=None: None)

    relay.close()

    assert events == ["mapped", "unmapped 7", "destroyed"]


def test_a_mapping_that_torch_cannot_wrap_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[int] = []

    def refuse_to_wrap(pointer: int, num_bytes: int, device_index: int) -> None:
        raise RuntimeError("DLPack import failed")

    mappings = PeerPoolMappings(
        level_zero=SimpleNamespace(
            open_peer_segment=lambda blob, descriptor, device_index: 7,
            address_range=lambda pointer: AddressRange(
                base_pointer=7, segment_bytes=POOL_BYTES
            ),
            close_peer_segment=closed.append,
        ),
        wait_executor=ThreadPoolExecutor(max_workers=1),
        engine_id="receiver",
    )
    monkeypatch.setattr(
        level_zero_ipc_peers,
        "fetch_peer_segment",
        lambda socket_name, pool_id: (b"", os.open("/dev/null", os.O_RDONLY)),
    )
    monkeypatch.setattr(level_zero_ipc_peers, "wrap_device_bytes", refuse_to_wrap)
    reference = peer_reference(segment_offset_bytes=0, pool_bytes=POOL_BYTES)

    with pytest.raises(RuntimeError, match="DLPack import failed"):
        mappings.map_peer_pool(reference, 0)
    assert closed == [7]
    mappings.wait_executor.shutdown()


@pytest.mark.parametrize(
    ("segment_offset_bytes", "pool_bytes"),
    [(-64, POOL_BYTES), (0, 2 * POOL_BYTES), (POOL_BYTES, 64), (0, 0)],
    ids=["negative-offset", "past-the-end", "offset-past-the-end", "empty"],
)
def test_a_peer_window_outside_the_mapping_is_refused_and_unmapped(
    monkeypatch: pytest.MonkeyPatch, segment_offset_bytes: int, pool_bytes: int
) -> None:
    """The window comes off the wire, and a wrong one would let torch read or
    write device memory past the mapping."""
    closed: list[int] = []
    mappings = PeerPoolMappings(
        level_zero=SimpleNamespace(
            open_peer_segment=lambda blob, descriptor, device_index: 7,
            address_range=lambda pointer: AddressRange(
                base_pointer=7, segment_bytes=POOL_BYTES
            ),
            close_peer_segment=closed.append,
        ),
        wait_executor=ThreadPoolExecutor(max_workers=1),
        engine_id="receiver",
    )
    monkeypatch.setattr(
        level_zero_ipc_peers,
        "fetch_peer_segment",
        lambda socket_name, pool_id: (b"", os.open("/dev/null", os.O_RDONLY)),
    )
    monkeypatch.setattr(
        level_zero_ipc_peers,
        "wrap_device_bytes",
        Mock(side_effect=AssertionError("an invalid window reached torch")),
    )
    reference = peer_reference(
        segment_offset_bytes=segment_offset_bytes, pool_bytes=pool_bytes
    )

    with pytest.raises(ValueError, match="claims bytes"):
        mappings.map_peer_pool(reference, 0)
    assert closed == [7]
    mappings.wait_executor.shutdown()


def test_every_peer_pool_is_unmapped_even_when_one_fails() -> None:
    unmapped: list[int] = []

    def close_peer_segment(pointer: int) -> None:
        if pointer == 7:
            raise RuntimeError("unmap failed")
        else:
            unmapped.append(pointer)

    mappings = PeerPoolMappings(
        level_zero=SimpleNamespace(close_peer_segment=close_peer_segment),
        wait_executor=ThreadPoolExecutor(max_workers=1),
        engine_id="receiver",
    )
    for pool_id, pointer in (("first", 7), ("second", 8)):
        mapping: Future[PeerPool] = Future()
        mapping.set_result(
            PeerPool(
                peer_pointer=pointer,
                pool_tensor=torch.empty(0),
                socket_name="sender-socket",
            )
        )
        mappings.mappings[pool_id] = mapping

    with pytest.raises(RuntimeError, match="unmap failed"):
        mappings.close()
    assert unmapped == [8]
    assert mappings.mappings == {}
    mappings.wait_executor.shutdown()


def test_a_transfer_past_a_cached_peer_mapping_is_refused() -> None:
    """The mapping was sized by the pool's first transfer, and a later message that
    claims a larger pool must not read past it: the slice would be cut short and
    copy_ could broadcast the remainder into the destination."""
    cached_mapping = torch.zeros(POOL_BYTES, dtype=torch.uint8)

    async def cached_pool(
        reference: PeerPoolReference, destination_device_index: int
    ) -> torch.Tensor:
        return cached_mapping

    released: list[str] = []
    relay = LevelZeroIpcRelay.__new__(LevelZeroIpcRelay)
    relay.engine_id = "receiver"
    relay.peer_pools = SimpleNamespace(
        pool_tensor_for=cached_pool, release_reader=released.append
    )
    destination = SimpleNamespace(
        device=SimpleNamespace(type="xpu", index=0),
        view=lambda dtype: torch.zeros(4096, dtype=torch.uint8),
    )
    slot_bytes = SLOT_SIZE_KB * 1024
    metadata = {
        "engine_id": "sender",
        "transfer_info": {
            "size": 4096,
            "offset": POOL_BYTES,
            "slot_index": POOL_BYTES // slot_bytes,
            "slot_size": slot_bytes,
            "num_slots": 1,
            "allocation_size": slot_bytes,
        },
        "level_zero_ipc": {
            "pool_id": "sender:1:abc",
            "socket_name": "sglang-omni-level-zero-1-abc",
            "segment_offset": 0,
            "pool_bytes": 4 * POOL_BYTES,
            "source_device_index": 0,
        },
    }

    with pytest.raises(ValueError, match="past the"):
        asyncio.run(relay.get_async(metadata, destination, request_id="r"))
    # Nothing was queued against the mapping, so it must not stay busy.
    assert released == ["sender:1:abc"]


def mapped_peer_pools(
    unmapped: list[int],
) -> PeerPoolMappings:
    """A receiver's mapping cache over a fake context that records unmaps."""
    return PeerPoolMappings(
        level_zero=SimpleNamespace(close_peer_segment=unmapped.append),
        wait_executor=ThreadPoolExecutor(max_workers=1),
        engine_id="receiver",
    )


def settle_pool(
    mappings: PeerPoolMappings,
    pool_id: str,
    *,
    peer_pointer: int,
    readers: int = 0,
    last_used_ns: int = 0,
) -> None:
    """Put an already mapped pool in the cache, as a finished mapping would."""
    mapping: Future[PeerPool] = Future()
    mapping.set_result(
        PeerPool(
            peer_pointer=peer_pointer,
            pool_tensor=torch.empty(0),
            socket_name=f"socket-{pool_id}",
            readers=readers,
            last_used_ns=last_used_ns,
        )
    )
    mappings.mappings[pool_id] = mapping


def test_a_pool_whose_sender_stopped_is_unmapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restarted sender publishes a new pool id, so without this the dead
    sender's pool stays mapped and its memory stays spent on its own card."""
    unmapped: list[int] = []
    mappings = mapped_peer_pools(unmapped)
    settle_pool(mappings, "gone", peer_pointer=7)
    settle_pool(mappings, "alive", peer_pointer=8)
    monkeypatch.setattr(
        level_zero_ipc_peers, "bound_rendezvous_names", lambda: {"socket-alive"}
    )

    mappings.release_spent_pools(keep_at_most=MAX_MAPPED_PEER_POOLS - 1)

    assert unmapped == [7]
    assert set(mappings.mappings) == {"alive"}
    mappings.wait_executor.shutdown()


def test_mapping_a_new_pool_releases_a_dead_sender_s_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing else runs on a receiver between transfers, so taking a new mapping
    is when stale ones have to go."""
    unmapped: list[int] = []
    mappings = mapped_peer_pools(unmapped)
    settle_pool(mappings, "gone", peer_pointer=7)
    monkeypatch.setattr(
        level_zero_ipc_peers,
        "bound_rendezvous_names",
        lambda: {"sglang-omni-level-zero-1-abc"},
    )
    monkeypatch.setattr(
        mappings,
        "map_peer_pool",
        lambda reference, destination_device_index: PeerPool(
            peer_pointer=9,
            pool_tensor=torch.empty(0),
            socket_name=reference.socket_name,
        ),
    )

    tensor = asyncio.run(mappings.pool_tensor_for(peer_reference_bytes(), 0))

    assert tensor is not None
    assert unmapped == [7]
    assert set(mappings.mappings) == {"sender:1:abc"}
    # The new mapping is counted as read until its copy drains.
    assert mappings.mapped_pool("sender:1:abc").readers == 1
    mappings.wait_executor.shutdown()


def test_a_pool_being_read_is_never_unmapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unmapping under a read would pull the memory its copy is still sourcing."""
    unmapped: list[int] = []
    mappings = mapped_peer_pools(unmapped)
    settle_pool(mappings, "busy", peer_pointer=7, readers=1)
    monkeypatch.setattr(level_zero_ipc_peers, "bound_rendezvous_names", lambda: set())

    mappings.release_spent_pools(keep_at_most=MAX_MAPPED_PEER_POOLS - 1)
    assert unmapped == []
    assert set(mappings.mappings) == {"busy"}

    # Once the read reports it has drained, the dead sender's pool can go.
    mappings.release_reader("busy")
    mappings.release_spent_pools(keep_at_most=MAX_MAPPED_PEER_POOLS - 1)
    assert unmapped == [7]
    mappings.wait_executor.shutdown()


def test_the_oldest_idle_pool_goes_when_the_cap_is_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unmapped: list[int] = []
    mappings = mapped_peer_pools(unmapped)
    for index in range(MAX_MAPPED_PEER_POOLS):
        settle_pool(
            mappings,
            f"pool-{index}",
            peer_pointer=100 + index,
            # pool-3 was used longest ago.
            last_used_ns=1000 - index if index != 3 else 1,
        )
    monkeypatch.setattr(
        level_zero_ipc_peers,
        "bound_rendezvous_names",
        lambda: {f"socket-pool-{index}" for index in range(MAX_MAPPED_PEER_POOLS)},
    )

    mappings.release_spent_pools(keep_at_most=MAX_MAPPED_PEER_POOLS - 1)

    assert unmapped == [103]
    assert len(mappings.mappings) == MAX_MAPPED_PEER_POOLS - 1
    mappings.wait_executor.shutdown()


def test_a_full_cache_of_busy_pools_is_reported_rather_than_unmapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unmapped: list[int] = []
    mappings = mapped_peer_pools(unmapped)
    for index in range(MAX_MAPPED_PEER_POOLS):
        settle_pool(mappings, f"pool-{index}", peer_pointer=100 + index, readers=1)
    monkeypatch.setattr(
        level_zero_ipc_peers,
        "bound_rendezvous_names",
        lambda: {f"socket-pool-{index}" for index in range(MAX_MAPPED_PEER_POOLS)},
    )

    mappings.release_spent_pools(keep_at_most=MAX_MAPPED_PEER_POOLS - 1)

    assert unmapped == []
    assert len(mappings.mappings) == MAX_MAPPED_PEER_POOLS
    mappings.wait_executor.shutdown()


def test_a_finished_read_reports_its_pool_is_free() -> None:
    """The Get operation is what tells the cache a mapping stopped being read."""
    released: list[str] = []

    async def run() -> None:
        event = SimpleNamespace(query=lambda: True)
        operation = LevelZeroIpcGetOperation(
            event=event,
            peer_pool=torch.empty(0),
            placement=TransferPlacement(
                size_bytes=8,
                offset_bytes=0,
                slot_index=0,
                slot_size_bytes=8,
                num_slots=1,
            ),
            device_index=0,
            wait_executor=ThreadPoolExecutor(max_workers=1),
            request_id="r",
            on_reader_done=lambda: released.append("pool"),
        )
        await operation.wait_for_completion(timeout=5)
        # Reported once, however many times completion is awaited.
        await operation.wait_for_completion(timeout=5)
        operation.wait_executor.shutdown()

    asyncio.run(run())
    assert released == ["pool"]


def peer_reference_bytes() -> PeerPoolReference:
    return peer_reference(segment_offset_bytes=0, pool_bytes=POOL_BYTES)


def test_a_live_rendezvous_is_the_only_one_named_as_bound() -> None:
    """Reading the table must not disturb the sender, so a live rendezvous is
    found without a connection it would have to answer."""
    descriptor = os.open(os.devnull, os.O_RDONLY)
    segment = ExportedSegment(
        base_pointer=0,
        segment_bytes=64,
        handle_blob=bytes(64),
        file_descriptor=descriptor,
    )
    rendezvous = HandleRendezvous("pool", segment)
    try:
        assert rendezvous.socket_name in level_zero_ipc_peers.bound_rendezvous_names()
    finally:
        rendezvous.close()
        os.close(descriptor)
    assert rendezvous.socket_name not in level_zero_ipc_peers.bound_rendezvous_names()


def test_an_unreadable_socket_table_unmaps_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing is provably gone without the table, so no pool may be released."""
    unmapped: list[int] = []
    mappings = mapped_peer_pools(unmapped)
    settle_pool(mappings, "pool", peer_pointer=7)
    monkeypatch.setattr(level_zero_ipc_peers, "bound_rendezvous_names", lambda: None)

    mappings.release_spent_pools(keep_at_most=MAX_MAPPED_PEER_POOLS - 1)

    assert unmapped == []
    assert set(mappings.mappings) == {"pool"}
    mappings.wait_executor.shutdown()


def test_a_timed_out_read_keeps_its_pool_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timeout does not prove the copy stopped reading the mapping, so the pool
    must stay held; close() synchronizes the device before it unmaps."""
    released: list[str] = []
    # The real boundary enters torch.xpu.device(0), which has no card to select
    # on a CPU or CUDA worker; the timeout is what this test is about.
    monkeypatch.setattr(
        level_zero_ipc_operations,
        "synchronize_xpu_event",
        lambda event, device_index: event.synchronize(),
    )

    async def run() -> None:
        operation = LevelZeroIpcGetOperation(
            event=SimpleNamespace(
                query=lambda: False, synchronize=lambda: time.sleep(5)
            ),
            peer_pool=torch.empty(0),
            placement=TransferPlacement(
                size_bytes=8,
                offset_bytes=0,
                slot_index=0,
                slot_size_bytes=8,
                num_slots=1,
            ),
            device_index=0,
            wait_executor=ThreadPoolExecutor(max_workers=1),
            request_id="r",
            on_reader_done=lambda: released.append("pool"),
        )
        with pytest.raises(asyncio.TimeoutError):
            await operation.wait_for_completion(timeout=0.05)
        assert released == []
        assert operation.completed is False
        operation.wait_executor.shutdown(wait=False)

    asyncio.run(run())


def test_a_pool_a_read_is_waiting_for_is_not_evicted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Awaiting an already mapped pool still yields, and the read is not counted
    until it resumes. A sweep in that window must not unmap the pool, or the read
    resumes with a tensor over memory that is gone.
    """
    unmapped: list[int] = []
    mappings = mapped_peer_pools(unmapped)
    settle_pool(mappings, "sender:1:abc", peer_pointer=7)
    # No sender name is bound, so the sweep would call this pool stale.
    monkeypatch.setattr(level_zero_ipc_peers, "bound_rendezvous_names", lambda: set())

    async def run() -> torch.Tensor:
        borrower = asyncio.create_task(
            mappings.pool_tensor_for(peer_reference_bytes(), 0)
        )
        # Let the borrower reach its await and suspend there.
        await asyncio.sleep(0)
        mappings.release_spent_pools(keep_at_most=MAX_MAPPED_PEER_POOLS - 1)
        return await borrower

    tensor = asyncio.run(run())

    assert unmapped == []
    assert tensor is mappings.mapped_pool("sender:1:abc").pool_tensor
    assert mappings.mapped_pool("sender:1:abc").readers == 1
    assert mappings.waiting_readers == {}
    mappings.wait_executor.shutdown()


def test_a_pool_still_being_mapped_is_not_evicted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mapping that has not finished opening has no pointer to unmap yet."""
    unmapped: list[int] = []
    mappings = mapped_peer_pools(unmapped)
    opening: Future[PeerPool] = Future()
    mappings.mappings["sender:1:abc"] = opening
    monkeypatch.setattr(level_zero_ipc_peers, "bound_rendezvous_names", lambda: set())

    mappings.release_spent_pools(keep_at_most=MAX_MAPPED_PEER_POOLS - 1)

    assert unmapped == []
    assert set(mappings.mappings) == {"sender:1:abc"}
    opening.cancel()
    mappings.wait_executor.shutdown()


def test_a_pool_that_will_not_unmap_is_kept_for_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forgetting a mapping whose unmap failed would leave the sender's pool
    mapped with nothing tracking it, so close() could never retry."""
    mappings = PeerPoolMappings(
        level_zero=SimpleNamespace(
            close_peer_segment=Mock(side_effect=RuntimeError("unmap failed"))
        ),
        wait_executor=ThreadPoolExecutor(max_workers=1),
        engine_id="receiver",
    )
    settle_pool(mappings, "stuck", peer_pointer=7)
    monkeypatch.setattr(level_zero_ipc_peers, "bound_rendezvous_names", lambda: set())

    assert mappings.unmap_pool("stuck", "its sender stopped serving") is False
    assert set(mappings.mappings) == {"stuck"}

    # The sweep stops rather than picking the same pool every round.
    mappings.release_spent_pools(keep_at_most=MAX_MAPPED_PEER_POOLS - 1)
    assert set(mappings.mappings) == {"stuck"}
    mappings.wait_executor.shutdown()


@pytest.mark.timeout(60)
def test_the_cap_sweep_stops_on_a_pool_that_will_not_unmap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The least recently used pool is chosen every round, so a sweep that does
    not stop on a failed unmap picks the same one forever."""
    attempts: list[int] = []

    def refuse_to_close(peer_pointer: int) -> None:
        attempts.append(peer_pointer)
        raise RuntimeError("unmap failed")

    mappings = PeerPoolMappings(
        level_zero=SimpleNamespace(close_peer_segment=refuse_to_close),
        wait_executor=ThreadPoolExecutor(max_workers=1),
        engine_id="receiver",
    )
    for index in range(MAX_MAPPED_PEER_POOLS):
        settle_pool(
            mappings, f"pool-{index}", peer_pointer=100 + index, last_used_ns=index + 1
        )
    # Every sender still serves, so only the cap branch can choose a pool.
    monkeypatch.setattr(
        level_zero_ipc_peers,
        "bound_rendezvous_names",
        lambda: {f"socket-pool-{index}" for index in range(MAX_MAPPED_PEER_POOLS)},
    )

    mappings.release_spent_pools(keep_at_most=MAX_MAPPED_PEER_POOLS - 1)

    # pool-0 is the oldest; it is tried once and the sweep gives up.
    assert attempts == [100]
    assert len(mappings.mappings) == MAX_MAPPED_PEER_POOLS
    mappings.wait_executor.shutdown()


def test_a_full_cache_of_busy_pools_admits_the_next_sender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cap bounds the pools no read holds. Refusing a transfer because other
    pools are being read would fail a request over something it has no part in,
    so the mapping is admitted and reported instead."""
    unmapped: list[int] = []
    mappings = mapped_peer_pools(unmapped)
    for index in range(MAX_MAPPED_PEER_POOLS):
        settle_pool(mappings, f"busy-{index}", peer_pointer=100 + index, readers=1)
    monkeypatch.setattr(level_zero_ipc_peers, "bound_rendezvous_names", lambda: set())
    monkeypatch.setattr(
        mappings,
        "map_peer_pool",
        lambda reference, device_index: PeerPool(
            peer_pointer=9, pool_tensor=torch.empty(0), socket_name="socket-new"
        ),
    )

    with caplog_at_warning() as records:
        tensor = asyncio.run(mappings.pool_tensor_for(peer_reference_bytes(), 0))

    assert tensor is not None
    assert unmapped == []
    assert len(mappings.mappings) == MAX_MAPPED_PEER_POOLS + 1
    assert any("every one is being read" in record.message for record in records)
    mappings.wait_executor.shutdown()


@contextlib.contextmanager
def caplog_at_warning():
    """Collect warnings the peers module logs, without a pytest fixture."""
    records: list[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Collector(level=logging.WARNING)
    logger = logging.getLogger(level_zero_ipc_peers.__name__)
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


def test_overflow_pools_go_as_soon_as_their_reads_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pools admitted while every mapping was busy are only reclaimable once
    their reads finish, and a run of transfers to pools already mapped never
    reaches the sweep that admitting a new one does."""
    unmapped: list[int] = []
    mappings = mapped_peer_pools(unmapped)
    over_cap = MAX_MAPPED_PEER_POOLS + 1
    for index in range(over_cap):
        settle_pool(
            mappings,
            f"pool-{index}",
            peer_pointer=100 + index,
            readers=1,
            last_used_ns=index + 1,
        )
    monkeypatch.setattr(
        level_zero_ipc_peers,
        "bound_rendezvous_names",
        lambda: {f"socket-pool-{index}" for index in range(over_cap)},
    )

    # Draining all but one leaves the cache over the cap by one.
    for index in range(over_cap - 1):
        mappings.release_reader(f"pool-{index}")

    # The oldest idle pool went; the one still being read stayed.
    assert unmapped == [100]
    assert len(mappings.mappings) == MAX_MAPPED_PEER_POOLS
    assert mappings.mapped_pool(f"pool-{over_cap - 1}").readers == 1
    mappings.wait_executor.shutdown()


def test_a_read_finishing_under_the_cap_does_not_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sweep reads /proc/net/unix, so the usual case must not reach it."""
    unmapped: list[int] = []
    mappings = mapped_peer_pools(unmapped)
    settle_pool(mappings, "pool", peer_pointer=7, readers=1)

    def refuse_to_read_the_table() -> set[str]:
        raise AssertionError("the socket table was read while under the cap")

    monkeypatch.setattr(
        level_zero_ipc_peers, "bound_rendezvous_names", refuse_to_read_the_table
    )

    mappings.release_reader("pool")

    assert unmapped == []
    assert mappings.mapped_pool("pool").readers == 0
    mappings.wait_executor.shutdown()


def test_capturing_the_layout_reports_it_on_a_successful_acquire() -> None:
    """capture_layout exists to explain pool pressure, so the success path has to
    report the layout it acquired against, not a placeholder."""

    async def run() -> None:
        allocator = ContiguousSlotAllocator(slot_count=8, slot_size=8)
        # Free a hole of 2 and a hole of 3, so the numbers are distinguishable.
        held = await allocator.acquire_async(8)
        allocator.release(held.offset, 8)
        taken = await allocator.acquire_async(2)
        middle = await allocator.acquire_async(1)
        allocator.release(taken.offset, 2)

        allocation = await allocator.acquire_async(1, capture_layout=True)

        # Two runs free before this acquire: the 2 given back and the tail of 5.
        assert allocation.free_slots_before == 7
        assert allocation.free_runs_before == 2
        assert allocation.largest_free_run_before == 5
        # Without capture_layout the fields stay unset rather than guessed.
        plain = await allocator.acquire_async(1)
        assert plain.free_runs_before == -1
        assert plain.largest_free_run_before == -1
        allocator.release(middle.offset, 1)

    asyncio.run(run())


def test_overlapping_waits_report_one_read_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two waits on one read both resume after the event. Reporting twice would
    drop a reader that another read still holds, and that pool could then be
    unmapped while its copy runs."""
    released: list[str] = []

    async def wait_without_a_device(event, *, device_index, wait_executor, timeout):
        """The real wait enters torch.xpu.device(0); only the timing matters here."""
        if event.query():
            return
        else:
            await asyncio.wait_for(asyncio.to_thread(event.synchronize), timeout)

    monkeypatch.setattr(
        level_zero_ipc_operations, "wait_for_xpu_event", wait_without_a_device
    )

    async def run() -> None:
        finished = threading.Event()
        operation = LevelZeroIpcGetOperation(
            event=SimpleNamespace(
                query=lambda: False,
                synchronize=lambda: finished.wait(PEER_TIMEOUT_S),
            ),
            peer_pool=torch.empty(0),
            placement=TransferPlacement(
                size_bytes=8,
                offset_bytes=0,
                slot_index=0,
                slot_size_bytes=8,
                num_slots=1,
            ),
            device_index=0,
            wait_executor=ThreadPoolExecutor(max_workers=2),
            request_id="r",
            on_reader_done=lambda: released.append("pool"),
        )
        waits = [
            asyncio.create_task(operation.wait_for_completion(timeout=PEER_TIMEOUT_S))
            for _ in range(2)
        ]
        # Both are past the completed check and inside the wait before it ends.
        await asyncio.sleep(0.05)
        finished.set()
        await asyncio.gather(*waits)
        operation.wait_executor.shutdown()

    asyncio.run(run())
    assert released == ["pool"]


def test_overlapping_put_waits_release_the_slots_once() -> None:
    """The allocator raises on a double release, so completion has to be
    idempotent even when two waits resume together."""

    async def run() -> None:
        allocator = ContiguousSlotAllocator(slot_count=2, slot_size=8)
        allocation = await allocator.acquire_async(1)
        operation = LevelZeroIpcPutOperation(
            {},
            on_failure=lambda error: None,
            allocator=allocator,
            placement=TransferPlacement(
                size_bytes=8,
                offset_bytes=allocation.offset,
                slot_index=0,
                slot_size_bytes=8,
                num_slots=1,
            ),
            request_id="r",
        )
        waits = [
            asyncio.create_task(operation.wait_for_completion(timeout=PEER_TIMEOUT_S))
            for _ in range(2)
        ]
        await asyncio.sleep(0)
        operation.mark_receiver_done()
        await asyncio.gather(*waits)
        # Released exactly once: every slot free again, and no "released twice".
        assert allocator.free_slots == allocator.slot_count

    asyncio.run(run())


def test_a_mapping_that_fails_to_open_and_to_unmap_is_kept_for_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cache drops a failed mapping, so a pointer that will not unmap has
    nothing holding it; close() is the only thing left that can try again."""
    closed: list[int] = []
    attempts: list[int] = []

    def close_peer_segment(peer_pointer: int) -> None:
        attempts.append(peer_pointer)
        if len(attempts) == 1:
            raise RuntimeError("unmap failed")
        else:
            closed.append(peer_pointer)

    mappings = PeerPoolMappings(
        level_zero=SimpleNamespace(
            open_peer_segment=lambda blob, descriptor, device_index: 7,
            address_range=lambda pointer: SimpleNamespace(
                base_pointer=7, segment_bytes=POOL_BYTES
            ),
            close_peer_segment=close_peer_segment,
        ),
        wait_executor=ThreadPoolExecutor(max_workers=1),
        engine_id="receiver",
    )
    monkeypatch.setattr(
        level_zero_ipc_peers,
        "fetch_peer_segment",
        lambda socket_name, pool_id: (b"", os.open(os.devnull, os.O_RDONLY)),
    )
    # A window the mapping cannot satisfy, so opening fails after the import.
    reference = peer_reference(segment_offset_bytes=0, pool_bytes=4 * POOL_BYTES)

    with pytest.raises(ValueError, match="claims bytes"):
        mappings.map_peer_pool(reference, 0)

    assert mappings.pointers_still_mapped == [7]
    mappings.close()
    assert closed == [7]
    assert mappings.pointers_still_mapped == []
    mappings.wait_executor.shutdown()


def test_a_mapping_that_fails_after_its_reader_was_cancelled_is_forgotten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation skips the handler that drops a failed mapping, because
    CancelledError is a BaseException. Left cached, the next read would replay
    the old failure instead of importing again."""
    may_fail = threading.Event()
    attempts: list[int] = []

    def map_then_fail(reference: PeerPoolReference, device_index: int) -> PeerPool:
        attempts.append(device_index)
        assert may_fail.wait(PEER_TIMEOUT_S)
        raise RuntimeError("the sender went away")

    mappings = mapped_peer_pools([])
    monkeypatch.setattr(mappings, "map_peer_pool", map_then_fail)
    # The cache belongs to the event loop, and the worker thread is what
    # completes the mapping, so the eviction has to be handed back to the loop.
    mappings.mappings = ThreadRecordingCache(mappings.mappings)

    async def run() -> None:
        reader = asyncio.create_task(
            mappings.pool_tensor_for(peer_reference_bytes(), 0)
        )
        await asyncio.sleep(0)
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
        # The worker is shielded, so it is still opening at this point.
        assert set(mappings.mappings) == {"sender:1:abc"}

        may_fail.set()
        # Let the worker finish and its callback reach the loop.
        for _ in range(100):
            await asyncio.sleep(0.01)
            if not mappings.mappings:
                break
            else:
                pass
        assert mappings.mappings == {}
        assert mappings.waiting_readers == {}
        assert mappings.mappings.removal_threads == [threading.get_ident()]

        # A later read imports again rather than replaying the old failure.
        with pytest.raises(RuntimeError, match="the sender went away"):
            await mappings.pool_tensor_for(peer_reference_bytes(), 0)
        assert len(attempts) == 2

    asyncio.run(run())
    mappings.wait_executor.shutdown()


class ThreadRecordingCache(dict):
    """A mapping cache that records which thread removed each entry."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.removal_threads: list[int] = []

    def pop(self, *args: object, **kwargs: object) -> object:
        removed = super().pop(*args, **kwargs)
        self.removal_threads.append(threading.get_ident())
        return removed
