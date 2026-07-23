# SPDX-License-Identifier: Apache-2.0
"""CUDA IPC relay backed by a bounded sender-side GPU slot pool."""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, NamedTuple

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor

from sglang_omni.profiler.comm_trace import elapsed_ms as _comm_elapsed_ms
from sglang_omni.profiler.comm_trace import emit as _comm_trace
from sglang_omni.profiler.comm_trace import enabled as _comm_trace_enabled
from sglang_omni.profiler.comm_trace import now_ns as _comm_now_ns

from .base import Relay, RelayOperation, register_relay

logger = logging.getLogger(__name__)

_PEER_ENABLED: set[tuple[int, int]] = set()
_PEER_VISIBILITY_WARNED: set[tuple[int, int, int]] = set()
_DEFAULT_WAIT_THREADS = 8


class _CudaEventWaitResult(NamedTuple):
    worker_start_ns: int
    worker_done_ns: int


def _event_wait_threads_from_env() -> int:
    value = os.getenv("SGLANG_OMNI_CUDA_IPC_WAIT_THREADS")
    if value is None:
        return _DEFAULT_WAIT_THREADS
    threads = int(value)
    if threads <= 0:
        raise ValueError("SGLANG_OMNI_CUDA_IPC_WAIT_THREADS must be positive")
    return threads


def _synchronize_cuda_event(
    event: torch.cuda.Event,
    device_index: int,
) -> _CudaEventWaitResult:
    worker_start_ns = _comm_now_ns()
    with torch.cuda.device(device_index):
        event.synchronize()
    return _CudaEventWaitResult(
        worker_start_ns=worker_start_ns,
        worker_done_ns=_comm_now_ns(),
    )


def _cuda_event_elapsed_ms(
    start_event: torch.cuda.Event | None,
    end_event: torch.cuda.Event | None,
) -> float | None:
    if start_event is None or end_event is None:
        return None
    try:
        return float(start_event.elapsed_time(end_event))
    except RuntimeError:
        return None


def _parse_device_id(device: str) -> int:
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    return 0


def _ensure_peer_access(src_index: int, dst_index: int) -> None:
    """Enable P2P access when available and warn when it is not."""
    if src_index == dst_index:
        return
    key = (dst_index, src_index)
    if key in _PEER_ENABLED:
        return
    if not torch.cuda.can_device_access_peer(dst_index, src_index):
        logger.warning(
            "cuda_ipc: GPU %d cannot peer-access GPU %d; cross-GPU copy will "
            "stage through host memory (no NVLink fast path)",
            dst_index,
            src_index,
        )
    _PEER_ENABLED.add(key)


def _dump_cuda_storage_handle(tensor: torch.Tensor) -> dict[str, Any]:
    (
        storage_device,
        storage_handle,
        storage_size_bytes,
        storage_offset_bytes,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    ) = tensor.untyped_storage()._share_cuda_()
    return {
        "storage_device": int(storage_device),
        "storage_handle": storage_handle,
        "storage_size_bytes": int(storage_size_bytes),
        "storage_offset_bytes": int(storage_offset_bytes),
        "ref_counter_handle": ref_counter_handle,
        "ref_counter_offset": int(ref_counter_offset),
        "event_handle": event_handle,
        "event_sync_required": bool(event_sync_required),
        "numel": int(tensor.numel()),
    }


def _load_cuda_storage_handle(
    storage_meta: dict[str, Any],
    *,
    device: torch.device,
) -> torch.Tensor:
    device_index = int(device.index or 0)
    return rebuild_cuda_tensor(
        torch.Tensor,
        (int(storage_meta["numel"]),),
        (1,),
        0,
        torch.UntypedStorage,
        torch.uint8,
        device_index,
        storage_meta["storage_handle"],
        int(storage_meta["storage_size_bytes"]),
        int(storage_meta["storage_offset_bytes"]),
        False,
        storage_meta["ref_counter_handle"],
        int(storage_meta["ref_counter_offset"]),
        storage_meta["event_handle"],
        bool(storage_meta["event_sync_required"]),
    )


def _slots_for_size(size: int, slot_size: int) -> int:
    if size < 0:
        raise ValueError("cuda_ipc transfer size must be non-negative")
    return max(1, (size + slot_size - 1) // slot_size)


class _SlotLayout(NamedTuple):
    slot_index: int | None
    free_slots: int
    largest_free_run: int
    free_runs: int


class _SlotAllocation(NamedTuple):
    offset: int
    wait_rounds: int
    free_slots_before: int
    largest_free_run_before: int
    free_runs_before: int
    last_failed_free_slots: int
    last_failed_largest_free_run: int
    last_failed_free_runs: int


class CudaIpcPutOperation(RelayOperation):
    """Sender-side handle; completion means the slot can be reused."""

    def __init__(
        self,
        metadata: dict[str, Any],
        *,
        ready_event: torch.cuda.Event,
        source_tensor: torch.Tensor,
        slot_index: int,
        request_id: str | None,
        size: int,
        release_cb: Callable[[], None],
        fail_cb: Callable[[BaseException], None],
        num_slots: int = 1,
        copy_start_event: torch.cuda.Event | None = None,
        copy_done_event: torch.cuda.Event | None = None,
    ) -> None:
        self._metadata = metadata
        self._ready_event: torch.cuda.Event | None = ready_event
        self._copy_start_event = copy_start_event
        self._copy_done_event = copy_done_event
        self._source_tensor: torch.Tensor | None = source_tensor
        self._slot_index = slot_index
        self._num_slots = num_slots
        self._request_id = request_id
        self._size = size
        self._release_cb = release_cb
        self._fail_cb = fail_cb
        self._completed = False
        self._receiver_done = asyncio.get_running_loop().create_future()
        self._receiver_done_mark_ns: int | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self._completed:
            return
        wait_start = _comm_now_ns()
        try:
            await asyncio.wait_for(self._receiver_done, timeout=timeout)
        except TimeoutError as exc:
            self._completed = True
            self._fail_cb(exc)
            self._source_tensor = None
            self._ready_event = None
            self._copy_start_event = None
            self._copy_done_event = None
            raise
        except Exception as exc:
            self._completed = True
            self._fail_cb(exc)
            self._source_tensor = None
            self._ready_event = None
            self._copy_start_event = None
            self._copy_done_event = None
            raise
        self._completed = True
        self._release_cb()
        ack_resume_ms = -1.0
        if self._receiver_done_mark_ns is not None:
            ack_resume_ms = _comm_elapsed_ms(self._receiver_done_mark_ns)
        sender_copy_gpu_ms = _cuda_event_elapsed_ms(
            self._copy_start_event, self._copy_done_event
        )
        self._source_tensor = None
        self._ready_event = None
        self._copy_start_event = None
        self._copy_done_event = None
        trace_fields: dict[str, Any] = {
            "request_id": self._request_id,
            "slot_index": self._slot_index,
            "num_slots": self._num_slots,
            "bytes": self._size,
            "elapsed_ms": round(_comm_elapsed_ms(wait_start), 6),
            "ack_resume_ms": round(ack_resume_ms, 6),
        }
        if sender_copy_gpu_ms is not None:
            trace_fields["sender_copy_gpu_ms"] = round(sender_copy_gpu_ms, 6)
        _comm_trace("cuda_ipc_put_wait_ack", **trace_fields)

    def mark_receiver_done(self) -> None:
        if not self._receiver_done.done():
            self._receiver_done_mark_ns = _comm_now_ns()
            self._receiver_done.set_result(None)

    def mark_receiver_failed(self, exc: BaseException) -> None:
        if not self._receiver_done.done():
            self._receiver_done.set_exception(exc)


class CudaIpcGetOperation(RelayOperation):
    """Receiver-side handle. Completion means the peer copy finished."""

    def __init__(
        self,
        event: torch.cuda.Event,
        pool_tensor: torch.Tensor,
        slot_index: int,
        num_slots: int,
        request_id: str | None,
        size: int,
        device_index: int,
        wait_executor: ThreadPoolExecutor,
        start_event: torch.cuda.Event | None = None,
        done_event: torch.cuda.Event | None = None,
    ) -> None:
        self._event = event
        self._start_event = start_event
        self._done_event = done_event
        self._pool_tensor: torch.Tensor | None = pool_tensor
        self._slot_index = slot_index
        self._num_slots = num_slots
        self._request_id = request_id
        self._size = size
        self._device_index = device_index
        self._wait_executor = wait_executor
        self._completed = False

    @property
    def metadata(self) -> Any:
        return None

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self._completed:
            return
        wait_start = _comm_now_ns()
        if self._event.query():
            host_wait_ms = _comm_elapsed_ms(wait_start)
            receiver_gpu_ms = _cuda_event_elapsed_ms(
                self._start_event, self._done_event
            )
            self._completed = True
            self._pool_tensor = None
            self._start_event = None
            self._done_event = None
            trace_fields: dict[str, Any] = {
                "request_id": self._request_id,
                "slot_index": self._slot_index,
                "num_slots": self._num_slots,
                "bytes": self._size,
                "completion_mode": "query_ready",
                "elapsed_ms": round(host_wait_ms, 6),
            }
            if receiver_gpu_ms is not None:
                trace_fields["receiver_gpu_wait_copy_ms"] = round(receiver_gpu_ms, 6)
                trace_fields["host_minus_receiver_gpu_ms"] = round(
                    host_wait_ms - receiver_gpu_ms, 6
                )
            _comm_trace("cuda_ipc_get_wait_copy", **trace_fields)
            return

        loop = asyncio.get_event_loop()
        submit_ns = _comm_now_ns()
        wait_future = loop.run_in_executor(
            self._wait_executor,
            _synchronize_cuda_event,
            self._event,
            self._device_index,
        )
        try:
            wait_result = await asyncio.wait_for(wait_future, timeout=timeout)
        except TimeoutError:
            self._completed = True
            self._pool_tensor = None
            self._start_event = None
            self._done_event = None
            raise
        except Exception:
            self._completed = True
            self._pool_tensor = None
            self._start_event = None
            self._done_event = None
            raise

        host_wait_ms = _comm_elapsed_ms(wait_start)
        receiver_gpu_ms = _cuda_event_elapsed_ms(self._start_event, self._done_event)
        self._completed = True
        self._pool_tensor = None
        self._start_event = None
        self._done_event = None
        worker_queue_ms = (wait_result.worker_start_ns - submit_ns) / 1_000_000.0
        worker_block_ms = (
            wait_result.worker_done_ns - wait_result.worker_start_ns
        ) / 1_000_000.0
        worker_done_to_resume_ms = _comm_elapsed_ms(wait_result.worker_done_ns)
        trace_fields: dict[str, Any] = {
            "request_id": self._request_id,
            "slot_index": self._slot_index,
            "num_slots": self._num_slots,
            "bytes": self._size,
            "completion_mode": "thread_synchronize",
            "worker_queue_ms": round(worker_queue_ms, 6),
            "worker_block_ms": round(worker_block_ms, 6),
            "worker_done_to_resume_ms": round(worker_done_to_resume_ms, 6),
            "elapsed_ms": round(host_wait_ms, 6),
        }
        if receiver_gpu_ms is not None:
            trace_fields["receiver_gpu_wait_copy_ms"] = round(receiver_gpu_ms, 6)
            trace_fields["host_minus_receiver_gpu_ms"] = round(
                host_wait_ms - receiver_gpu_ms, 6
            )
        _comm_trace("cuda_ipc_get_wait_copy", **trace_fields)


class _ContiguousSlotAllocator:
    def __init__(self, *, slot_count: int, slot_size: int) -> None:
        if slot_count <= 0:
            raise ValueError("slot_count must be positive")
        if slot_size <= 0:
            raise ValueError("slot_size must be positive")
        self.slot_count = slot_count
        self.slot_size = slot_size
        self._free = [True] * slot_count
        self._free_slots = slot_count
        self._lock = asyncio.Lock()
        self._changed = asyncio.Event()
        self._changed.set()

    async def acquire_async(
        self, num_slots: int, *, capture_layout: bool = False
    ) -> _SlotAllocation:
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if num_slots > self.slot_count:
            raise ValueError(
                f"allocation requires {num_slots} slots, but pool has "
                f"{self.slot_count}"
            )

        wait_rounds = 0
        last_failed_free_slots = 0
        last_failed_largest_free_run = 0
        last_failed_free_runs = 0
        while True:
            async with self._lock:
                slot_index = self._find_contiguous(num_slots)
                free_slots_before = self._free_slots
                if slot_index is not None:
                    for index in range(slot_index, slot_index + num_slots):
                        self._free[index] = False
                    self._free_slots -= num_slots
                    if self._free_slots == 0:
                        self._changed.clear()
                    return _SlotAllocation(
                        offset=slot_index * self.slot_size,
                        wait_rounds=wait_rounds,
                        free_slots_before=free_slots_before,
                        largest_free_run_before=-1,
                        free_runs_before=-1,
                        last_failed_free_slots=last_failed_free_slots,
                        last_failed_largest_free_run=last_failed_largest_free_run,
                        last_failed_free_runs=last_failed_free_runs,
                    )
                if capture_layout:
                    layout = self._find_contiguous_with_layout(num_slots)
                    last_failed_free_slots = free_slots_before
                    last_failed_largest_free_run = layout.largest_free_run
                    last_failed_free_runs = layout.free_runs
                wait_rounds += 1
                self._changed.clear()
            await self._changed.wait()

    def release(self, offset: int, num_slots: int) -> None:
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if offset % self.slot_size != 0:
            raise ValueError("offset must be slot aligned")
        slot_index = offset // self.slot_size
        if slot_index < 0 or slot_index + num_slots > self.slot_count:
            raise ValueError("slot range is outside the pool")
        for index in range(slot_index, slot_index + num_slots):
            if self._free[index]:
                raise RuntimeError("cuda_ipc slot released twice")
        for index in range(slot_index, slot_index + num_slots):
            self._free[index] = True
        self._free_slots += num_slots
        self._changed.set()

    def _find_contiguous(self, num_slots: int) -> int | None:
        run_start = 0
        run_len = 0
        for index, is_free in enumerate(self._free):
            if is_free:
                if run_len == 0:
                    run_start = index
                run_len += 1
                if run_len == num_slots:
                    return run_start
            else:
                run_len = 0
        return None

    def _find_contiguous_with_layout(self, num_slots: int) -> _SlotLayout:
        run_start = 0
        run_len = 0
        free_slots = 0
        free_runs = 0
        largest_free_run = 0
        slot_index: int | None = None
        in_run = False
        for index, is_free in enumerate(self._free):
            if is_free:
                free_slots += 1
                if not in_run:
                    in_run = True
                    free_runs += 1
                    run_start = index
                    run_len = 0
                run_len += 1
                largest_free_run = max(largest_free_run, run_len)
                if slot_index is None and run_len >= num_slots:
                    slot_index = run_start
            else:
                in_run = False
                run_len = 0
        return _SlotLayout(
            slot_index=slot_index,
            free_slots=free_slots,
            largest_free_run=largest_free_run,
            free_runs=free_runs,
        )


@register_relay("cuda_ipc")
class CudaIpcRelay(Relay):
    def __init__(
        self,
        engine_id: str,
        device: str = "cuda",
        slot_size_mb: int | None = 512,
        credits: int | None = 2,
        slot_size_kb: int = 64,
        pool_size_mb: int | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            raise TypeError(
                f"unexpected cuda_ipc relay options: {', '.join(sorted(kwargs))}"
            )
        self.engine_id = engine_id
        if device == "cpu":
            raise ValueError(
                "cuda_ipc relay requires a CUDA device; got 'cpu'. Use the shm "
                "relay for host-memory stages."
            )
        self.device = device
        self.device_id = _parse_device_id(device)
        if pool_size_mb is None:
            legacy_slot_size_mb = 512 if slot_size_mb is None else int(slot_size_mb)
            legacy_credits = 2 if credits is None else int(credits)
            pool_size_mb = legacy_slot_size_mb * legacy_credits
        self.slot_size = int(slot_size_kb) * 1024
        if self.slot_size <= 0:
            raise ValueError("cuda_ipc slot_size_kb must be positive")
        requested_pool_size = int(pool_size_mb) * 1024 * 1024
        self.slot_count = requested_pool_size // self.slot_size
        self.pool_size = self.slot_count * self.slot_size
        self.credits = self.slot_count
        if requested_pool_size <= 0:
            raise ValueError("cuda_ipc pool_size_mb must be positive")
        if self.slot_count <= 0:
            raise ValueError("cuda_ipc pool size must fit at least one slot")

        self._pool_tensor: torch.Tensor | None = None
        self._pool_id: str | None = None
        self._pool_storage_handles: dict[str, dict[str, Any]] = {}
        self._allocator: _ContiguousSlotAllocator | None = None

        self._remote_pools: dict[str, torch.Tensor] = {}
        self._failed_error: BaseException | None = None
        self._failed_event = asyncio.Event()
        self._wait_executor = ThreadPoolExecutor(
            max_workers=_event_wait_threads_from_env(),
            thread_name_prefix=f"cuda-ipc-wait-{engine_id}",
        )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            logger.debug("CudaIpcRelay finalizer cleanup failed", exc_info=True)

    def _ensure_local_pool(self) -> None:
        if self._pool_tensor is not None:
            return
        start = _comm_now_ns()
        total_pool_bytes = self.slot_size * self.slot_count
        device = torch.device(self.device)
        logger.info(
            "[%s] Allocating CUDA-IPC pool: %.2f MB on %s (%d x %dB slots)",
            self.engine_id,
            total_pool_bytes / 1024**2,
            self.device,
            self.slot_count,
            self.slot_size,
        )
        with torch.cuda.device(device):
            self._pool_tensor = torch.empty(
                total_pool_bytes, dtype=torch.uint8, device=device
            )
        self._pool_id = f"{self.engine_id}:{os.getpid()}:{uuid.uuid4().hex}"
        self._allocator = _ContiguousSlotAllocator(
            slot_count=self.slot_count,
            slot_size=self.slot_size,
        )
        _comm_trace(
            "cuda_ipc_pool_alloc",
            engine_id=self.engine_id,
            device=self.device,
            slot_size=self.slot_size,
            slot_count=self.slot_count,
            credits=self.slot_count,
            total_pool_bytes=total_pool_bytes,
            elapsed_ms=round(_comm_elapsed_ms(start), 6),
        )

    def _local_pool_state(
        self,
    ) -> tuple[torch.Tensor, str, _ContiguousSlotAllocator]:
        self._ensure_local_pool()
        pool_tensor = self._pool_tensor
        pool_id = self._pool_id
        allocator = self._allocator
        if pool_tensor is None:
            raise RuntimeError("cuda_ipc local pool tensor was not initialized")
        if pool_id is None:
            raise RuntimeError("cuda_ipc local pool id was not initialized")
        if allocator is None:
            raise RuntimeError("cuda_ipc local credit allocator was not initialized")
        return pool_tensor, pool_id, allocator

    def _pool_storage_handle_for(
        self,
        pool_tensor: torch.Tensor,
        receiver_id: str,
    ) -> tuple[dict[str, Any], bool]:
        storage_handle = self._pool_storage_handles.get(receiver_id)
        if storage_handle is not None:
            return storage_handle, False

        # Each PyTorch CUDA storage export carries one consumer refcounter
        # token. Reuse an export only for the Stage relay cache that imports it.
        storage_handle = _dump_cuda_storage_handle(pool_tensor)
        self._pool_storage_handles[receiver_id] = storage_handle
        return storage_handle, True

    def _mark_failed(self, exc: BaseException) -> None:
        if self._failed_error is None:
            self._failed_error = exc
            self._failed_event.set()

    def _raise_if_failed(self) -> None:
        if self._failed_error is not None:
            raise RuntimeError("cuda_ipc relay failed") from self._failed_error

    async def _acquire_slots(
        self, allocator: _ContiguousSlotAllocator, num_slots: int
    ) -> _SlotAllocation:
        self._raise_if_failed()
        acquire_task = asyncio.create_task(
            allocator.acquire_async(num_slots, capture_layout=_comm_trace_enabled())
        )
        fail_task = asyncio.create_task(self._failed_event.wait())
        try:
            done, _ = await asyncio.wait(
                {acquire_task, fail_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if fail_task in done:
                if acquire_task in done:
                    allocator.release(acquire_task.result().offset, num_slots)
                self._raise_if_failed()
                raise RuntimeError("cuda_ipc relay failed")

            allocation = acquire_task.result()
            try:
                self._raise_if_failed()
            except Exception:
                allocator.release(allocation.offset, num_slots)
                raise
            return allocation
        finally:
            for task in (acquire_task, fail_task):
                if not task.done():
                    task.cancel()

    def _get_remote_pool(
        self,
        metadata: dict[str, Any],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        ipc_meta = metadata["cuda_ipc"]
        pool_id = ipc_meta["pool_id"]
        pool = self._remote_pools.get(pool_id)
        if pool is None:
            storage_meta = ipc_meta["pool_storage"]
            if not isinstance(storage_meta, dict):
                raise TypeError(
                    "cuda_ipc pool_storage metadata must be a dict, got "
                    f"{type(storage_meta).__name__}"
                )
            pool = _load_cuda_storage_handle(storage_meta, device=device)
            self._remote_pools[pool_id] = pool
        return pool

    async def put_async(
        self,
        tensor: torch.Tensor,
        request_id: str | None = None,
        dst_rank: int | None = None,
        receiver_id: str | None = None,
    ) -> CudaIpcPutOperation:
        self._raise_if_failed()
        if receiver_id is None:
            raise ValueError("cuda_ipc put requires a receiver identity")
        if not tensor.is_cuda:
            raise ValueError(
                "cuda_ipc relay can only transfer CUDA tensors; "
                f"got tensor on {tensor.device}"
            )
        pool_tensor, pool_id, allocator = self._local_pool_state()
        flat = tensor.contiguous().view(torch.uint8).reshape(-1)
        size = int(flat.numel())
        num_slots = _slots_for_size(size, self.slot_size)
        if num_slots > allocator.slot_count:
            raise ValueError(
                f"Tensor size {size} requires {num_slots} cuda_ipc slots, "
                f"but pool has {allocator.slot_count}"
            )

        acquire_start = _comm_now_ns()
        allocation = await self._acquire_slots(allocator, num_slots)
        acquire_ms = _comm_elapsed_ms(acquire_start)
        offset = allocation.offset
        slot_index = int(offset // self.slot_size)

        try:
            copy_start = _comm_now_ns()
            pool_slice = pool_tensor[offset : offset + size]
            device = torch.device(self.device)
            stream = torch.cuda.current_stream(device)
            ready_event = torch.cuda.Event(interprocess=True)
            trace_timing = _comm_trace_enabled()
            copy_start_event = (
                torch.cuda.Event(enable_timing=True) if trace_timing else None
            )
            copy_done_event = (
                torch.cuda.Event(enable_timing=True) if trace_timing else None
            )
            with torch.cuda.device(device), torch.cuda.stream(stream):
                if copy_start_event is not None:
                    copy_start_event.record(stream)
                pool_slice.copy_(flat, non_blocking=True)
                if copy_done_event is not None:
                    copy_done_event.record(stream)
                ready_event.record(stream)
            copy_enqueue_ms = _comm_elapsed_ms(copy_start)
            handle_start = _comm_now_ns()
            ready_handle = ready_event.ipc_handle()
            handle_ms = _comm_elapsed_ms(handle_start)
            pool_export_start = _comm_now_ns()
            pool_storage_handle, pool_exported = self._pool_storage_handle_for(
                pool_tensor,
                receiver_id,
            )
            pool_export_ms = _comm_elapsed_ms(pool_export_start)
        except Exception:
            allocator.release(offset, num_slots)
            raise
        _comm_trace(
            "cuda_ipc_put_async",
            request_id=request_id,
            engine_id=self.engine_id,
            device=self.device,
            bytes=size,
            slot_index=slot_index,
            num_slots=num_slots,
            acquire_wait_rounds=allocation.wait_rounds,
            free_slots_before=allocation.free_slots_before,
            largest_free_run_before=allocation.largest_free_run_before,
            free_runs_before=allocation.free_runs_before,
            last_failed_free_slots=allocation.last_failed_free_slots,
            last_failed_largest_free_run=allocation.last_failed_largest_free_run,
            last_failed_free_runs=allocation.last_failed_free_runs,
            acquire_ms=round(acquire_ms, 6),
            copy_enqueue_ms=round(copy_enqueue_ms, 6),
            event_handle_ms=round(handle_ms, 6),
            pool_exported=pool_exported,
            pool_export_ms=round(pool_export_ms, 6),
        )

        metadata = {
            "engine_id": self.engine_id,
            "transfer_info": {
                "size": size,
                "offset": int(offset),
                "slot_index": slot_index,
                "slot_size": self.slot_size,
                "num_slots": num_slots,
                "allocation_size": num_slots * self.slot_size,
            },
            "cuda_ipc": {
                "pool_id": pool_id,
                "pool_storage": pool_storage_handle,
                "src_device_id": self.device_id,
                "ready_event": ready_handle,
            },
        }
        return CudaIpcPutOperation(
            metadata,
            ready_event=ready_event,
            source_tensor=flat,
            slot_index=slot_index,
            num_slots=num_slots,
            request_id=request_id,
            size=size,
            release_cb=lambda: allocator.release(offset, num_slots),
            fail_cb=self._mark_failed,
            copy_start_event=copy_start_event,
            copy_done_event=copy_done_event,
        )

    async def get_async(
        self,
        metadata: dict[str, Any],
        dest_tensor: torch.Tensor,
        request_id: str | None = None,
    ) -> CudaIpcGetOperation:
        if not dest_tensor.is_cuda:
            raise ValueError(
                "cuda_ipc relay can only receive into CUDA tensors; "
                f"dest is on {dest_tensor.device}"
            )
        start = _comm_now_ns()
        ipc_meta = metadata["cuda_ipc"]
        dst_device = dest_tensor.device
        pool_start = _comm_now_ns()
        pool_tensor = self._get_remote_pool(metadata, device=dst_device)
        pool_ms = _comm_elapsed_ms(pool_start)

        src_index = int(ipc_meta["src_device_id"])
        dst_index = int(dst_device.index or 0)
        peer_start = _comm_now_ns()
        device_count = torch.cuda.device_count()
        if 0 <= src_index < device_count:
            _ensure_peer_access(src_index, dst_index)
        else:
            warn_key = (dst_index, src_index, device_count)
            if warn_key not in _PEER_VISIBILITY_WARNED:
                _PEER_VISIBILITY_WARNED.add(warn_key)
                logger.warning(
                    "cuda_ipc source device %d is outside this receiver's visible "
                    "CUDA device range [0, %d); peer-access validation skipped. "
                    "This is expected only when sender and receiver use different "
                    "CUDA_VISIBLE_DEVICES namespaces.",
                    src_index,
                    device_count,
                )
        peer_ms = _comm_elapsed_ms(peer_start)

        size = int(metadata["transfer_info"]["size"])
        offset = int(metadata["transfer_info"]["offset"])
        slot_index = int(metadata["transfer_info"]["slot_index"])
        slot_size = int(metadata["transfer_info"]["slot_size"])
        num_slots = int(metadata["transfer_info"]["num_slots"])
        if offset < 0:
            raise ValueError("cuda_ipc transfer offset must be non-negative")
        if slot_size <= 0 or num_slots <= 0:
            raise ValueError("cuda_ipc slot_size and num_slots must be positive")
        if offset % slot_size != 0:
            raise ValueError("cuda_ipc transfer offset must be slot aligned")
        if num_slots < _slots_for_size(size, slot_size):
            raise ValueError("cuda_ipc num_slots is too small for transfer size")
        allocation_size = num_slots * slot_size
        if offset + allocation_size > int(pool_tensor.numel()):
            raise ValueError("cuda_ipc allocation range exceeds pool size")
        # Import on the waiting device; source-device imports can hang cross-GPU.
        event_start = _comm_now_ns()
        ready_event = torch.cuda.Event.from_ipc_handle(
            dst_device, ipc_meta["ready_event"]
        )
        event_ms = _comm_elapsed_ms(event_start)

        copy_start = _comm_now_ns()
        src = pool_tensor.view(torch.uint8).reshape(-1)[offset : offset + size]
        dst = dest_tensor.view(torch.uint8).reshape(-1)
        if dst.numel() < size:
            raise ValueError(
                f"cuda_ipc destination buffer has {dst.numel()} bytes, "
                f"but transfer requires {size} bytes"
            )
        copy_len = size

        stream = torch.cuda.current_stream(dst_device)
        trace_timing = _comm_trace_enabled()
        start_event = torch.cuda.Event(enable_timing=True) if trace_timing else None
        done_event = torch.cuda.Event(enable_timing=True) if trace_timing else None
        with torch.cuda.device(dst_device), torch.cuda.stream(stream):
            if start_event is not None:
                start_event.record(stream)
            stream.wait_event(ready_event)
            dst[:copy_len].copy_(src[:copy_len], non_blocking=True)
            if done_event is not None:
                done_event.record(stream)
        event = torch.cuda.Event()
        event.record(stream)
        copy_enqueue_ms = _comm_elapsed_ms(copy_start)
        _comm_trace(
            "cuda_ipc_get_async",
            request_id=request_id,
            engine_id=self.engine_id,
            src_device=src_index,
            dst_device=dst_index,
            bytes=size,
            copy_len=int(copy_len),
            slot_index=slot_index,
            num_slots=num_slots,
            pool_open_ms=round(pool_ms, 6),
            peer_access_ms=round(peer_ms, 6),
            event_import_ms=round(event_ms, 6),
            copy_enqueue_ms=round(copy_enqueue_ms, 6),
            elapsed_ms=round(_comm_elapsed_ms(start), 6),
        )
        return CudaIpcGetOperation(
            event,
            pool_tensor,
            slot_index,
            num_slots,
            request_id=request_id,
            size=size,
            device_index=dst_index,
            wait_executor=self._wait_executor,
            start_event=start_event,
            done_event=done_event,
        )

    def cleanup(self, request_id: str) -> None:
        pass

    def close(self) -> None:
        self._remote_pools.clear()
        self._pool_storage_handles.clear()
        self._pool_tensor = None
        self._allocator = None
        self._wait_executor.shutdown(wait=False, cancel_futures=True)
