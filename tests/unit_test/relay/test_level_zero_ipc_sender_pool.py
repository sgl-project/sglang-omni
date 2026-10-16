# SPDX-License-Identifier: Apache-2.0
"""How the Level-Zero IPC sender pool comes up, fails to, and goes away."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import sglang_omni.relay.level_zero_ipc as level_zero_ipc
from sglang_omni.platforms.level_zero import ExportedSegment
from sglang_omni.relay.level_zero_ipc import LevelZeroIpcRelay
from sglang_omni.relay.level_zero_rendezvous import HANDLE_REQUEST_LIMIT_BYTES
from sglang_omni.relay.slot_pool import ContiguousSlotAllocator

STAGED_BYTES = 4096


def unallocated_sender(
    engine_id: str, level_zero: SimpleNamespace
) -> LevelZeroIpcRelay:
    """A sender relay that has not made its pool yet, over a fake Level-Zero context."""
    relay = LevelZeroIpcRelay.__new__(LevelZeroIpcRelay)
    relay.engine_id = engine_id
    relay.device = "xpu:0"
    relay.device_index = 0
    relay.pool_size = 1 << 20
    relay.slot_size = 64 * 1024
    relay.slot_count = 16
    relay.pool_tensor = None
    relay.pool_pointer = None
    relay.pool_id = None
    relay.exported_segment = None
    relay.rendezvous = None
    relay.allocator = None
    relay.level_zero = level_zero
    return relay


def wrap_on_device(pointer: int, num_bytes: int, device_index: int) -> SimpleNamespace:
    return SimpleNamespace(device=SimpleNamespace(index=device_index))


def exported_at(pointer: int) -> ExportedSegment:
    return ExportedSegment(
        base_pointer=pointer, segment_bytes=64, handle_blob=b"", file_descriptor=-1
    )


def test_close_attempts_every_step_and_raises_the_first_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A step close skips is never retried, so its memory, handle or context would
    outlive the relay."""
    events: list[str] = []

    def refuse_to_unmap() -> None:
        raise RuntimeError("unmap failed")

    def refuse_to_release(segment: ExportedSegment) -> None:
        raise RuntimeError("release failed")

    relay = unallocated_sender(
        "sender",
        SimpleNamespace(
            release_segment=refuse_to_release,
            free=lambda pointer: events.append(f"freed {pointer}"),
            destroy=lambda: events.append("destroyed"),
        ),
    )
    relay.wait_executor = ThreadPoolExecutor(max_workers=1)
    relay.peer_pools = SimpleNamespace(close=refuse_to_unmap)
    relay.rendezvous = SimpleNamespace(close=lambda: events.append("rendezvous closed"))
    relay.exported_segment = exported_at(4096)
    relay.pool_pointer = 4096
    monkeypatch.setattr(torch.xpu, "synchronize", lambda device=None: None)

    with pytest.raises(RuntimeError, match="unmap failed"):
        relay.close()
    assert events == ["rendezvous closed", "freed 4096", "destroyed"]
    assert relay.pool_pointer is None
    assert relay.rendezvous is None


@pytest.mark.parametrize("failing_step", ["wrap", "export", "rendezvous"])
def test_a_pool_that_fails_to_come_up_is_freed_every_time(
    monkeypatch: pytest.MonkeyPatch, failing_step: str
) -> None:
    """A failed setup keeps nothing, so the next send allocates again; each attempt
    must give back what it allocated and exported."""
    events: list[str] = []
    pointers = iter((4096, 8192))

    def fail_at(step: str) -> None:
        if step == failing_step:
            raise RuntimeError(f"{step} failed")
        else:
            pass

    def wrap(pointer: int, num_bytes: int, device_index: int) -> SimpleNamespace:
        fail_at("wrap")
        return wrap_on_device(pointer, num_bytes, device_index)

    def export_segment(pointer: int) -> ExportedSegment:
        fail_at("export")
        return exported_at(pointer)

    def rendezvous(pool_id: str, segment: ExportedSegment) -> None:
        fail_at("rendezvous")
        raise AssertionError("every parametrized step fails before this returns")

    relay = unallocated_sender(
        "sender",
        SimpleNamespace(
            allocate_exportable=lambda device_index, num_bytes: next(pointers),
            export_segment=export_segment,
            release_segment=lambda segment: events.append(
                f"released {segment.base_pointer}"
            ),
            free=lambda pointer: events.append(f"freed {pointer}"),
        ),
    )
    monkeypatch.setattr(level_zero_ipc, "wrap_device_bytes", wrap)
    monkeypatch.setattr(level_zero_ipc, "HandleRendezvous", rendezvous)

    for _attempt in range(2):
        with pytest.raises(RuntimeError, match=f"{failing_step} failed"):
            relay.local_pool_state()

    exported = failing_step == "rendezvous"
    assert events == [
        *(["released 4096"] if exported else []),
        "freed 4096",
        *(["released 8192"] if exported else []),
        "freed 8192",
    ]
    assert relay.pool_pointer is None
    assert relay.exported_segment is None
    assert relay.rendezvous is None


def test_a_pool_is_freed_even_when_releasing_its_export_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freed: list[int] = []

    def refuse_to_release(segment: ExportedSegment) -> None:
        raise RuntimeError("release failed")

    def refuse_to_serve(pool_id: str, segment: ExportedSegment) -> None:
        raise RuntimeError("rendezvous failed")

    relay = unallocated_sender(
        "sender",
        SimpleNamespace(
            allocate_exportable=lambda device_index, num_bytes: 4096,
            export_segment=exported_at,
            release_segment=refuse_to_release,
            free=freed.append,
        ),
    )
    monkeypatch.setattr(level_zero_ipc, "wrap_device_bytes", wrap_on_device)
    monkeypatch.setattr(level_zero_ipc, "HandleRendezvous", refuse_to_serve)

    with pytest.raises(RuntimeError, match="release failed"):
        relay.local_pool_state()
    assert freed == [4096]


def test_a_pool_id_fits_the_rendezvous_request_whatever_the_engine_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served: list[str] = []

    def serve(pool_id: str, segment: ExportedSegment) -> SimpleNamespace:
        served.append(pool_id)
        return SimpleNamespace(socket_name="unused")

    relay = unallocated_sender(
        "talker_ar_" * 100,
        SimpleNamespace(
            allocate_exportable=lambda device_index, num_bytes: 4096,
            export_segment=exported_at,
        ),
    )
    monkeypatch.setattr(level_zero_ipc, "wrap_device_bytes", wrap_on_device)
    monkeypatch.setattr(level_zero_ipc, "HandleRendezvous", serve)

    relay.local_pool_state()

    assert len(served[0].encode("utf-8")) <= HANDLE_REQUEST_LIMIT_BYTES


async def no_wait(
    event: SimpleNamespace,
    *,
    device_index: int,
    wait_executor: ThreadPoolExecutor,
    timeout: float,
) -> None:
    """The staging copy into a host tensor has already finished."""


def staging_sender(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LevelZeroIpcRelay, ContiguousSlotAllocator]:
    """A sender whose pool is up, staging into a host tensor with the XPU calls
    stubbed out, so each one can be made to fail on its own."""
    allocator = ContiguousSlotAllocator(slot_count=4, slot_size=STAGED_BYTES)
    relay = unallocated_sender("sender", SimpleNamespace())
    relay.failed_error = None
    relay.failed_event = asyncio.Event()
    relay.segment_offset = 0
    relay.wait_executor = ThreadPoolExecutor(max_workers=1)
    relay.slot_size = STAGED_BYTES
    rendezvous = SimpleNamespace(socket_name="unused")
    pool = torch.zeros(4 * STAGED_BYTES, dtype=torch.uint8)
    relay.local_pool_state = lambda: (pool, "pool", allocator, rendezvous)
    monkeypatch.setattr(torch.xpu, "current_stream", lambda device=None: None)
    monkeypatch.setattr(
        torch.xpu, "Event", lambda: SimpleNamespace(record=lambda stream: None)
    )
    monkeypatch.setattr(torch.xpu, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.xpu, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(level_zero_ipc, "wait_for_xpu_event", no_wait)
    return relay, allocator


def xpu_payload() -> SimpleNamespace:
    """A payload the relay takes for an XPU tensor; its bytes live on the host."""
    host_bytes = torch.ones(STAGED_BYTES, dtype=torch.uint8)
    return SimpleNamespace(
        device=SimpleNamespace(type="xpu"), contiguous=lambda: host_bytes
    )


@pytest.mark.parametrize("failing_call", ["current_stream", "Event"])
def test_a_send_takes_no_slots_when_its_stream_or_event_cannot_be_made(
    monkeypatch: pytest.MonkeyPatch, failing_call: str
) -> None:
    relay, allocator = staging_sender(monkeypatch)
    monkeypatch.setattr(
        torch.xpu,
        failing_call,
        Mock(side_effect=RuntimeError(f"{failing_call} failed")),
    )

    with pytest.raises(RuntimeError, match=f"{failing_call} failed"):
        asyncio.run(relay.put_async(xpu_payload(), request_id="r"))
    assert allocator.free_slots == allocator.slot_count
    assert relay.failed_error is None
    relay.wait_executor.shutdown()


def test_slots_go_back_when_a_staged_transfer_cannot_be_published(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay, allocator = staging_sender(monkeypatch)
    monkeypatch.setattr(
        level_zero_ipc,
        "LevelZeroIpcPutOperation",
        Mock(side_effect=RuntimeError("publish failed")),
    )

    with pytest.raises(RuntimeError, match="publish failed"):
        asyncio.run(relay.put_async(xpu_payload(), request_id="r"))
    assert allocator.free_slots == allocator.slot_count
    assert relay.failed_error is None
    relay.wait_executor.shutdown()


def test_a_staged_transfer_keeps_its_slots_until_the_receiver_is_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay, allocator = staging_sender(monkeypatch)

    async def stage_and_finish() -> tuple[int, int]:
        operation = await relay.put_async(xpu_payload(), request_id="r")
        held = allocator.free_slots
        operation.mark_receiver_done()
        await operation.wait_for_completion()
        return held, allocator.free_slots

    held, released = asyncio.run(stage_and_finish())
    assert held == allocator.slot_count - 1
    assert released == allocator.slot_count
    relay.wait_executor.shutdown()


def test_slots_stay_reserved_when_the_queued_copy_cannot_be_tracked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the copy is queued it may still be writing the slots, and without its
    event nothing says when it stops, so handing them out again would let it
    overwrite a later payload."""
    relay, allocator = staging_sender(monkeypatch)

    def refuse_to_record(stream: None) -> None:
        raise RuntimeError("record failed")

    monkeypatch.setattr(
        torch.xpu, "Event", lambda: SimpleNamespace(record=refuse_to_record)
    )

    with pytest.raises(RuntimeError, match="record failed"):
        asyncio.run(relay.put_async(xpu_payload(), request_id="r"))
    assert allocator.free_slots == allocator.slot_count - 1
    assert isinstance(relay.failed_error, RuntimeError)
    relay.wait_executor.shutdown()
