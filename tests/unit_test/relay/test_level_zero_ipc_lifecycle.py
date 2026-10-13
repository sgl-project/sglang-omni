# SPDX-License-Identifier: Apache-2.0
"""What the Level-Zero IPC relay keeps or gives back when a transfer goes wrong."""
from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import sglang_omni.relay.level_zero_ipc as level_zero_ipc
import sglang_omni.relay.level_zero_ipc_peers as level_zero_ipc_peers
from sglang_omni.comm.engine import CommEngine
from sglang_omni.comm.router import CommRouter
from sglang_omni.platforms.level_zero import AddressRange
from sglang_omni.relay.level_zero_ipc import LevelZeroIpcRelay
from sglang_omni.relay.level_zero_ipc_metadata import (
    PeerPoolReference,
    TransferPlacement,
)
from sglang_omni.relay.level_zero_ipc_operations import LevelZeroIpcPutOperation
from sglang_omni.relay.level_zero_ipc_peers import PeerPool, PeerPoolMappings
from sglang_omni.relay.slot_pool import ContiguousSlotAllocator

SLOT_SIZE_KB = 64
POOL_SIZE_MB = 8
POOL_BYTES = 262144


def peer_reference(*, segment_offset: int, pool_bytes: int) -> PeerPoolReference:
    return PeerPoolReference(
        pool_id="sender:1:abc",
        socket_name="sglang-omni-level-zero-1-abc",
        segment_offset=segment_offset,
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
        return PeerPool(peer_pointer=7, pool_tensor=torch.empty(0))

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
    reference = peer_reference(segment_offset=0, pool_bytes=POOL_BYTES)

    with pytest.raises(RuntimeError, match="DLPack import failed"):
        mappings.map_peer_pool(reference, 0)
    assert closed == [7]
    mappings.wait_executor.shutdown()


@pytest.mark.parametrize(
    ("segment_offset", "pool_bytes"),
    [(-64, POOL_BYTES), (0, 2 * POOL_BYTES), (POOL_BYTES, 64), (0, 0)],
    ids=["negative-offset", "past-the-end", "offset-past-the-end", "empty"],
)
def test_a_peer_window_outside_the_mapping_is_refused_and_unmapped(
    monkeypatch: pytest.MonkeyPatch, segment_offset: int, pool_bytes: int
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
    reference = peer_reference(segment_offset=segment_offset, pool_bytes=pool_bytes)

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
        mapping.set_result(PeerPool(peer_pointer=pointer, pool_tensor=torch.empty(0)))
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

    relay = LevelZeroIpcRelay.__new__(LevelZeroIpcRelay)
    relay.engine_id = "receiver"
    relay.peer_pools = SimpleNamespace(pool_tensor_for=cached_pool)
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
