# SPDX-License-Identifier: Apache-2.0
"""Level-Zero IPC relay: card-to-card stage transfers out of a sender slot pool."""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

import torch

from sglang_omni.platforms.level_zero import ExportedSegment, LevelZeroContext
from sglang_omni.platforms.xpu_dlpack import wrap_device_bytes
from sglang_omni.profiler.comm_trace import elapsed_ms as comm_trace_elapsed_ms
from sglang_omni.profiler.comm_trace import emit as comm_trace_emit
from sglang_omni.profiler.comm_trace import enabled as comm_trace_enabled
from sglang_omni.profiler.comm_trace import now_ns as comm_trace_now_ns
from sglang_omni.relay.base import Relay, register_relay
from sglang_omni.relay.level_zero_ipc_metadata import (
    LevelZeroIpcMetadata,
    TransferPlacement,
    parse_metadata,
)
from sglang_omni.relay.level_zero_ipc_operations import (
    LevelZeroIpcGetOperation,
    LevelZeroIpcPutOperation,
    wait_for_xpu_event,
)
from sglang_omni.relay.level_zero_ipc_peers import PeerPoolMappings, ensure_peer_access
from sglang_omni.relay.level_zero_rendezvous import HandleRendezvous
from sglang_omni.relay.slot_pool import (
    ContiguousSlotAllocator,
    SlotAllocation,
    slots_for_size,
)

logger = logging.getLogger(__name__)

DEFAULT_WAIT_THREADS = 8
WAIT_THREADS_ENV = "SGLANG_OMNI_LEVEL_ZERO_IPC_WAIT_THREADS"

# The staging copy waits only on the sender's own work, so this means a stall.
STAGE_COPY_TIMEOUT_S = 30.0


def wait_thread_count() -> int:
    value = os.getenv(WAIT_THREADS_ENV)
    if value is None:
        return DEFAULT_WAIT_THREADS
    else:
        pass
    threads = int(value)
    if threads <= 0:
        raise ValueError(f"{WAIT_THREADS_ENV} must be positive")
    else:
        pass
    return threads


@register_relay("level_zero_ipc")
class LevelZeroIpcRelay(Relay):
    def __init__(
        self,
        engine_id: str,
        device: str,
        slot_size_kb: int,
        pool_size_mb: int,
    ) -> None:
        self.engine_id = engine_id
        parsed_device = torch.device(device)
        if parsed_device.type != "xpu":
            raise ValueError(
                f"level_zero_ipc relay requires an XPU device; got {device!r}. Use the "
                f"shm relay for host-memory stages."
            )
        else:
            pass
        self.device = device
        self.device_index = 0 if parsed_device.index is None else parsed_device.index
        self.slot_size = int(slot_size_kb) * 1024
        if self.slot_size <= 0:
            raise ValueError("level_zero_ipc slot_size_kb must be positive")
        else:
            pass
        requested_pool_size = int(pool_size_mb) * 1024 * 1024
        if requested_pool_size <= 0:
            raise ValueError("level_zero_ipc pool_size_mb must be positive")
        else:
            pass
        self.slot_count = requested_pool_size // self.slot_size
        if self.slot_count <= 0:
            raise ValueError("level_zero_ipc pool size must fit at least one slot")
        else:
            pass
        self.pool_size = self.slot_count * self.slot_size
        # Read first, since nothing destroys a context that a later check abandons.
        wait_threads = wait_thread_count()

        self.level_zero = LevelZeroContext()
        self.pool_pointer: int | None = None
        self.pool_tensor: torch.Tensor | None = None
        self.pool_id: str | None = None
        self.rendezvous: HandleRendezvous | None = None
        self.exported_segment: ExportedSegment | None = None
        self.segment_offset = 0
        self.allocator: ContiguousSlotAllocator | None = None
        self.failed_error: BaseException | None = None
        self.failed_event = asyncio.Event()
        self.wait_executor = ThreadPoolExecutor(
            max_workers=wait_threads,
            thread_name_prefix=f"level-zero-ipc-wait-{engine_id}",
        )
        self.peer_pools = PeerPoolMappings(
            level_zero=self.level_zero,
            wait_executor=self.wait_executor,
            engine_id=engine_id,
        )

    def local_pool_state(
        self,
    ) -> tuple[torch.Tensor, str, ContiguousSlotAllocator, HandleRendezvous]:
        """Allocate, export, and publish the sender pool on first use."""
        if self.pool_tensor is None:
            start = comm_trace_now_ns()
            pool_pointer = self.level_zero.allocate_exportable(
                self.device_index, self.pool_size
            )
            segment: ExportedSegment | None = None
            try:
                pool_tensor = wrap_device_bytes(
                    pool_pointer, self.pool_size, self.device_index
                )
                if pool_tensor.device.index != self.device_index:
                    raise RuntimeError(
                        f"level_zero_ipc pool allocated on xpu:{self.device_index} "
                        f"surfaced in torch as {pool_tensor.device}"
                    )
                else:
                    pass
                allocator = ContiguousSlotAllocator(
                    slot_count=self.slot_count,
                    slot_size=self.slot_size,
                )
                segment = self.level_zero.export_segment(pool_pointer)
                pool_id = f"{os.getpid()}:{uuid.uuid4().hex}"
                rendezvous = HandleRendezvous(pool_id, segment)
            except BaseException:
                # The next send would allocate a new pool, so this one goes now.
                try:
                    if segment is not None:
                        self.level_zero.release_segment(segment)
                    else:
                        pass
                finally:
                    self.level_zero.free(pool_pointer)
                raise
            self.pool_pointer = pool_pointer
            self.segment_offset = pool_pointer - segment.base_pointer
            self.pool_id = pool_id
            self.exported_segment = segment
            self.rendezvous = rendezvous
            self.allocator = allocator
            self.pool_tensor = pool_tensor
            logger.info(
                f"[{self.engine_id}] Allocated Level-Zero IPC pool: "
                f"{self.pool_size / 1024**2:.2f} MB on {self.device} "
                f"({self.slot_count} x {self.slot_size}B slots)"
            )
            comm_trace_emit(
                "level_zero_ipc_pool_alloc",
                engine_id=self.engine_id,
                device=self.device,
                slot_size=self.slot_size,
                slot_count=self.slot_count,
                total_pool_bytes=self.pool_size,
                elapsed_ms=round(comm_trace_elapsed_ms(start), 6),
            )
        else:
            pass

        pool_tensor = self.pool_tensor
        pool_id = self.pool_id
        allocator = self.allocator
        rendezvous = self.rendezvous
        if (
            pool_tensor is None
            or pool_id is None
            or allocator is None
            or rendezvous is None
        ):
            raise RuntimeError("level_zero_ipc local pool was not initialized")
        else:
            pass
        return pool_tensor, pool_id, allocator, rendezvous

    def mark_failed(self, error: BaseException) -> None:
        if self.failed_error is None:
            self.failed_error = error
            self.failed_event.set()
        else:
            pass

    def raise_if_failed(self) -> None:
        if self.failed_error is not None:
            raise RuntimeError("level_zero_ipc relay failed") from self.failed_error
        else:
            pass

    async def acquire_slots(
        self, allocator: ContiguousSlotAllocator, num_slots: int
    ) -> SlotAllocation:
        """Wait for pool space, giving up as soon as the relay fails."""
        self.raise_if_failed()
        acquire_task = asyncio.create_task(
            allocator.acquire_async(num_slots, capture_layout=comm_trace_enabled())
        )
        failure_task = asyncio.create_task(self.failed_event.wait())
        handed_over = False
        try:
            await asyncio.wait(
                {acquire_task, failure_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            self.raise_if_failed()
            allocation = acquire_task.result()
            handed_over = True
            return allocation
        finally:
            for task in (acquire_task, failure_task):
                if not task.done():
                    task.cancel()
                else:
                    pass
            # Slots granted to a caller that failed or was cancelled go back.
            if (
                not handed_over
                and acquire_task.done()
                and not acquire_task.cancelled()
                and acquire_task.exception() is None
            ):
                allocator.release(acquire_task.result().offset, num_slots)
            else:
                pass

    async def put_async(
        self,
        tensor: torch.Tensor,
        request_id: str | None = None,
        dst_rank: int | None = None,
        receiver_id: str | None = None,
    ) -> LevelZeroIpcPutOperation:
        self.raise_if_failed()
        if tensor.device.type != "xpu":
            raise ValueError(
                f"level_zero_ipc relay can only transfer XPU tensors; got tensor on "
                f"{tensor.device}"
            )
        else:
            pass
        pool_tensor, pool_id, allocator, rendezvous = self.local_pool_state()
        flat = tensor.contiguous().view(torch.uint8).reshape(-1)
        size = int(flat.numel())
        num_slots = slots_for_size(size, self.slot_size)
        if num_slots > allocator.slot_count:
            raise ValueError(
                f"Tensor size {size} requires {num_slots} level_zero_ipc slots, but "
                f"pool has {allocator.slot_count}"
            )
        else:
            pass

        # Made before the slots are taken, since neither touches the pool.
        device = torch.device(self.device)
        stream = torch.xpu.current_stream(device)
        staged_event = torch.xpu.Event()
        acquire_start = comm_trace_now_ns()
        allocation = await self.acquire_slots(allocator, num_slots)
        acquire_ms = comm_trace_elapsed_ms(acquire_start)
        offset = allocation.offset
        placement = TransferPlacement(
            size_bytes=size,
            offset_bytes=int(offset),
            slot_index=int(offset // self.slot_size),
            slot_size_bytes=self.slot_size,
            num_slots=num_slots,
        )
        copy_start = comm_trace_now_ns()
        try:
            with torch.xpu.device(device), torch.xpu.stream(stream):
                pool_tensor[offset : offset + size].copy_(flat, non_blocking=True)
                staged_event.record(stream)
            # The receiver has no XPU event to wait on, so the bytes land first.
            await wait_for_xpu_event(
                staged_event,
                device_index=self.device_index,
                wait_executor=self.wait_executor,
                timeout=STAGE_COPY_TIMEOUT_S,
            )
        except BaseException as error:
            # The copy may still write these slots, so they stay reserved.
            self.mark_failed(error)
            raise
        copy_ms = comm_trace_elapsed_ms(copy_start)
        try:
            operation = self.publish_staged_transfer(
                placement,
                pool_id=pool_id,
                rendezvous=rendezvous,
                allocator=allocator,
                request_id=request_id,
            )
        except BaseException:
            # The copy has drained and nothing was published, so the slots go back.
            allocator.release(offset, num_slots)
            raise
        comm_trace_emit(
            "level_zero_ipc_put_async",
            request_id=request_id,
            engine_id=self.engine_id,
            device=self.device,
            receiver_id=receiver_id,
            bytes=size,
            slot_index=placement.slot_index,
            num_slots=num_slots,
            acquire_wait_rounds=allocation.wait_rounds,
            free_slots_before=allocation.free_slots_before,
            acquire_ms=round(acquire_ms, 6),
            stage_copy_ms=round(copy_ms, 6),
        )
        return operation

    def publish_staged_transfer(
        self,
        placement: TransferPlacement,
        *,
        pool_id: str,
        rendezvous: HandleRendezvous,
        allocator: ContiguousSlotAllocator,
        request_id: str | None,
    ) -> LevelZeroIpcPutOperation:
        """Describe a payload that has reached the pool, for the receiver to read."""
        metadata: LevelZeroIpcMetadata = {
            "engine_id": self.engine_id,
            "transfer_info": {
                "size": placement.size_bytes,
                "offset": placement.offset_bytes,
                "slot_index": placement.slot_index,
                "slot_size": placement.slot_size_bytes,
                "num_slots": placement.num_slots,
                "allocation_size": placement.num_slots * placement.slot_size_bytes,
            },
            "level_zero_ipc": {
                "pool_id": pool_id,
                "socket_name": rendezvous.socket_name,
                "segment_offset": int(self.segment_offset),
                "pool_bytes": int(self.pool_size),
                "source_device_index": self.device_index,
            },
        }
        return LevelZeroIpcPutOperation(
            metadata,
            on_failure=self.mark_failed,
            allocator=allocator,
            placement=placement,
            request_id=request_id,
        )

    async def get_async(
        self,
        metadata: Mapping[str, object],
        dest_tensor: torch.Tensor,
        request_id: str | None = None,
    ) -> LevelZeroIpcGetOperation:
        if dest_tensor.device.type != "xpu":
            raise ValueError(
                f"level_zero_ipc relay can only receive into XPU tensors; dest is on "
                f"{dest_tensor.device}"
            )
        else:
            pass
        start = comm_trace_now_ns()
        placement, reference = parse_metadata(metadata)
        destination_device_index = int(dest_tensor.device.index or 0)
        ensure_peer_access(reference.source_device_index, destination_device_index)

        destination = dest_tensor.view(torch.uint8).reshape(-1)
        if destination.numel() < placement.size_bytes:
            raise ValueError(
                f"level_zero_ipc destination buffer has {destination.numel()} bytes, "
                f"but transfer requires {placement.size_bytes} bytes"
            )
        else:
            pass

        peer_pool = await self.peer_pools.pool_tensor_for(
            reference, destination_device_index
        )
        # The cached mapping is sized by the pool's first transfer; nothing widens it.
        allocation_end = (
            placement.offset_bytes + placement.num_slots * placement.slot_size_bytes
        )
        if allocation_end > peer_pool.numel():
            raise ValueError(
                f"level_zero_ipc transfer ends at byte {allocation_end}, past the "
                f"{peer_pool.numel()} bytes mapped for pool {reference.pool_id}"
            )
        else:
            pass
        source = peer_pool[
            placement.offset_bytes : placement.offset_bytes + placement.size_bytes
        ]
        stream = torch.xpu.current_stream(dest_tensor.device)
        event = torch.xpu.Event()
        with torch.xpu.device(dest_tensor.device), torch.xpu.stream(stream):
            destination[: placement.size_bytes].copy_(source, non_blocking=True)
            event.record(stream)
        comm_trace_emit(
            "level_zero_ipc_get_async",
            request_id=request_id,
            engine_id=self.engine_id,
            source_device=reference.source_device_index,
            destination_device=destination_device_index,
            bytes=placement.size_bytes,
            slot_index=placement.slot_index,
            num_slots=placement.num_slots,
            elapsed_ms=round(comm_trace_elapsed_ms(start), 6),
        )
        return LevelZeroIpcGetOperation(
            event=event,
            peer_pool=peer_pool,
            placement=placement,
            device_index=destination_device_index,
            wait_executor=self.wait_executor,
            request_id=request_id,
        )

    def cleanup(self, request_id: str) -> None:
        pass

    def close(self) -> None:
        # Drain the device and the mapping workers first; a failed step skips none.
        first_error: Exception | None = None
        for step in (
            lambda: torch.xpu.synchronize(self.device_index),
            lambda: self.wait_executor.shutdown(wait=True),
            self.peer_pools.close,
            self.close_rendezvous,
            self.release_exported_segment,
            self.free_pool,
            self.level_zero.destroy,
        ):
            try:
                step()
            except Exception as error:
                if first_error is None:
                    first_error = error
                else:
                    logger.warning(
                        f"[{self.engine_id}] Level-Zero IPC teardown step also "
                        f"failed: {error}"
                    )
        if first_error is not None:
            raise first_error
        else:
            pass

    def close_rendezvous(self) -> None:
        if self.rendezvous is not None:
            self.rendezvous.close()
            self.rendezvous = None
        else:
            pass

    def release_exported_segment(self) -> None:
        if self.exported_segment is not None:
            self.level_zero.release_segment(self.exported_segment)
            self.exported_segment = None
        else:
            pass

    def free_pool(self) -> None:
        self.pool_tensor = None
        if self.pool_pointer is not None:
            # A peer's mapping keeps the memory alive on its own.
            self.level_zero.free(self.pool_pointer)
            self.pool_pointer = None
        else:
            pass
