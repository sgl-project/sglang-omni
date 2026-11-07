# SPDX-License-Identifier: Apache-2.0
"""Sender and receiver operations of one Level-Zero IPC transfer."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import torch

from sglang_omni.profiler.comm_trace import elapsed_ms as comm_trace_elapsed_ms
from sglang_omni.profiler.comm_trace import emit as comm_trace_emit
from sglang_omni.profiler.comm_trace import now_ns as comm_trace_now_ns
from sglang_omni.relay.base import RelayOperation
from sglang_omni.relay.level_zero_ipc_metadata import (
    LevelZeroIpcMetadata,
    TransferPlacement,
)
from sglang_omni.relay.slot_pool import ContiguousSlotAllocator


def synchronize_xpu_event(event: torch.xpu.Event, device_index: int) -> None:
    with torch.xpu.device(device_index):
        event.synchronize()


async def wait_for_xpu_event(
    event: torch.xpu.Event,
    *,
    device_index: int,
    wait_executor: ThreadPoolExecutor,
    timeout: float,
) -> None:
    """Wait for an XPU event without blocking the asyncio event loop."""
    if event.query():
        return
    else:
        pass
    wait_future = asyncio.get_running_loop().run_in_executor(
        wait_executor,
        synchronize_xpu_event,
        event,
        device_index,
    )
    await asyncio.wait_for(wait_future, timeout=timeout)


class RelayFailureCallback(Protocol):
    """Fails the whole relay, so no later transfer reuses slots still being read."""

    def __call__(self, error: BaseException) -> None: ...


class LevelZeroIpcPutOperation(RelayOperation):
    """Sender-side handle; completion means the staged slots can be reused."""

    def __init__(
        self,
        metadata: LevelZeroIpcMetadata,
        *,
        on_failure: RelayFailureCallback,
        allocator: ContiguousSlotAllocator,
        placement: TransferPlacement,
        request_id: str | None,
    ) -> None:
        self.staged_metadata = metadata
        self.on_failure = on_failure
        self.allocator = allocator
        self.placement = placement
        self.request_id = request_id
        self.receiver_done = asyncio.get_running_loop().create_future()
        self.completed = False

    @property
    def metadata(self) -> LevelZeroIpcMetadata:
        return self.staged_metadata

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self.completed:
            return
        else:
            pass
        wait_start = comm_trace_now_ns()
        try:
            await asyncio.wait_for(self.receiver_done, timeout=timeout)
        except BaseException as error:
            self.strand_slots(error)
            raise
        self.completed = True
        self.allocator.release(self.placement.offset_bytes, self.placement.num_slots)
        comm_trace_emit(
            "level_zero_ipc_put_wait_ack",
            request_id=self.request_id,
            slot_index=self.placement.slot_index,
            num_slots=self.placement.num_slots,
            bytes=self.placement.size_bytes,
            elapsed_ms=round(comm_trace_elapsed_ms(wait_start), 6),
        )

    def record_sender_cancellation(self, error: BaseException) -> None:
        """The engine stopped waiting for the ACK without calling wait_for_completion."""
        if not self.completed:
            self.strand_slots(error)
        else:
            pass

    def strand_slots(self, error: BaseException) -> None:
        """Keep the slots reserved and fail the relay: the receiver may still read them."""
        self.completed = True
        comm_trace_emit(
            "level_zero_ipc_slots_stranded",
            request_id=self.request_id,
            slot_index=self.placement.slot_index,
            num_slots=self.placement.num_slots,
            bytes=self.placement.size_bytes,
            reason=type(error).__name__,
        )
        self.on_failure(error)

    def mark_receiver_done(self) -> None:
        if not self.receiver_done.done():
            self.receiver_done.set_result(None)
        else:
            pass

    def mark_receiver_failed(self, error: BaseException) -> None:
        if not self.receiver_done.done():
            self.receiver_done.set_exception(error)
        else:
            pass


class LevelZeroIpcGetOperation(RelayOperation):
    """Receiver-side handle; completion means the peer copy finished."""

    def __init__(
        self,
        *,
        event: torch.xpu.Event,
        peer_pool: torch.Tensor,
        placement: TransferPlacement,
        device_index: int,
        wait_executor: ThreadPoolExecutor,
        request_id: str | None,
    ) -> None:
        self.event = event
        self.peer_pool: torch.Tensor | None = peer_pool
        self.placement = placement
        self.device_index = device_index
        self.wait_executor = wait_executor
        self.request_id = request_id
        self.completed = False

    @property
    def metadata(self) -> None:
        return None

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self.completed:
            return
        else:
            pass
        wait_start = comm_trace_now_ns()
        try:
            await wait_for_xpu_event(
                self.event,
                device_index=self.device_index,
                wait_executor=self.wait_executor,
                timeout=timeout,
            )
        finally:
            self.completed = True
            self.peer_pool = None
        comm_trace_emit(
            "level_zero_ipc_get_wait_copy",
            request_id=self.request_id,
            slot_index=self.placement.slot_index,
            num_slots=self.placement.num_slots,
            bytes=self.placement.size_bytes,
            elapsed_ms=round(comm_trace_elapsed_ms(wait_start), 6),
        )
