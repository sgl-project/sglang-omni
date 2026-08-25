# SPDX-License-Identifier: Apache-2.0
"""CUDA IPC relay backed by a bounded sender-side GPU slot pool."""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import Any, Callable, NamedTuple

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor

from sglang_omni.comm.kv_transfer import KVPool
from sglang_omni.profiler.comm_trace import elapsed_ms as _comm_elapsed_ms
from sglang_omni.profiler.comm_trace import emit as _comm_trace
from sglang_omni.profiler.comm_trace import enabled as _comm_trace_enabled
from sglang_omni.profiler.comm_trace import now_ns as _comm_now_ns

from .base import Relay, RelayOperation, register_relay

logger = logging.getLogger(__name__)

_PEER_ENABLED: set[tuple[int, int]] = set()
_PEER_UNAVAILABLE: set[tuple[int, int]] = set()
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


async def _wait_for_cuda_event(
    event: torch.cuda.Event,
    *,
    device_index: int,
    wait_executor: ThreadPoolExecutor,
    timeout: float,
    finish_on_interrupt: bool = False,
) -> tuple[int | None, _CudaEventWaitResult | None]:
    """Wait for a CUDA event without blocking the asyncio event loop."""

    if event.query():
        return None, None
    loop = asyncio.get_running_loop()
    submit_ns = _comm_now_ns()
    wait_future = loop.run_in_executor(
        wait_executor,
        _synchronize_cuda_event,
        event,
        device_index,
    )
    if not finish_on_interrupt:
        return submit_ns, await asyncio.wait_for(wait_future, timeout=timeout)
    try:
        result = await asyncio.wait_for(asyncio.shield(wait_future), timeout=timeout)
    except (asyncio.CancelledError, TimeoutError):
        # A launched GPU copy cannot be cancelled. Keep its resources alive
        # until the kernel stops touching them, then propagate the interruption.
        await asyncio.shield(wait_future)
        raise
    return submit_ns, result


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


def _ensure_peer_access(src_index: int, dst_index: int) -> bool:
    """Check P2P access and report whether a cross-GPU copy is safe."""
    if src_index == dst_index:
        return True
    key = (dst_index, src_index)
    if key in _PEER_UNAVAILABLE:
        return False
    if key in _PEER_ENABLED:
        return True
    if not torch.cuda.can_device_access_peer(dst_index, src_index):
        _PEER_UNAVAILABLE.add(key)
        logger.warning(
            "cuda_ipc: GPU %d cannot peer-access GPU %d; CUDA IPC cannot perform "
            "this cross-GPU copy (select the shm relay instead)",
            dst_index,
            src_index,
        )
        return False
    _PEER_ENABLED.add(key)
    return True


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
        "tensor_offset": int(tensor.storage_offset()),
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
        int(storage_meta.get("tensor_offset", 0)),
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


class _ReceiverAckOperation(RelayOperation):
    """Common operation state for sender resources held until receiver ACK."""

    def __init__(
        self,
        metadata: dict[str, Any],
        *,
        held_references: tuple[Any, ...] = (),
    ) -> None:
        self._metadata = metadata
        self._receiver_done = asyncio.get_running_loop().create_future()
        self._receiver_done_mark_ns: int | None = None
        self._held_references = held_references
        self._completed = False

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    async def _wait_for_receiver(self, timeout: float) -> None:
        await asyncio.wait_for(self._receiver_done, timeout=timeout)

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self._completed:
            return
        try:
            await self._wait_for_receiver(timeout)
        finally:
            self._completed = True
            self._held_references = ()

    def mark_receiver_done(self) -> None:
        if not self._receiver_done.done():
            self._receiver_done_mark_ns = _comm_now_ns()
            self._receiver_done.set_result(None)

    def mark_receiver_failed(self, exc: BaseException) -> None:
        if not self._receiver_done.done():
            self._receiver_done.set_exception(exc)


class CudaIpcPutOperation(_ReceiverAckOperation):
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
        super().__init__(metadata)
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

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self._completed:
            return
        wait_start = _comm_now_ns()
        try:
            await self._wait_for_receiver(timeout)
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


class CudaIpcGetOperation(RelayOperation):
    """Receiver-side handle. Completion means the peer copy finished."""

    def __init__(
        self,
        event: torch.cuda.Event,
        pool_tensor: torch.Tensor | None,
        slot_index: int,
        num_slots: int,
        request_id: str | None,
        size: int,
        device_index: int,
        wait_executor: ThreadPoolExecutor,
        start_event: torch.cuda.Event | None = None,
        done_event: torch.cuda.Event | None = None,
        held_references: tuple[Any, ...] = (),
        finish_on_interrupt: bool = False,
        emit_trace: bool = True,
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
        self._held_references = held_references
        self._finish_on_interrupt = finish_on_interrupt
        self._emit_trace = emit_trace
        self._completed = False

    @property
    def metadata(self) -> Any:
        return None

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self._completed:
            return
        wait_start = _comm_now_ns()
        try:
            submit_ns, wait_result = await _wait_for_cuda_event(
                self._event,
                device_index=self._device_index,
                wait_executor=self._wait_executor,
                timeout=timeout,
                finish_on_interrupt=self._finish_on_interrupt,
            )
        except BaseException:
            self._release_references()
            raise

        host_wait_ms = _comm_elapsed_ms(wait_start)
        receiver_gpu_ms = _cuda_event_elapsed_ms(self._start_event, self._done_event)
        self._release_references()
        trace_fields: dict[str, Any] = {
            "request_id": self._request_id,
            "slot_index": self._slot_index,
            "num_slots": self._num_slots,
            "bytes": self._size,
            "completion_mode": (
                "query_ready" if wait_result is None else "thread_synchronize"
            ),
            "elapsed_ms": round(host_wait_ms, 6),
        }
        if submit_ns is not None and wait_result is not None:
            trace_fields.update(
                worker_queue_ms=round(
                    (wait_result.worker_start_ns - submit_ns) / 1_000_000.0, 6
                ),
                worker_block_ms=round(
                    (wait_result.worker_done_ns - wait_result.worker_start_ns)
                    / 1_000_000.0,
                    6,
                ),
                worker_done_to_resume_ms=round(
                    _comm_elapsed_ms(wait_result.worker_done_ns), 6
                ),
            )
        if receiver_gpu_ms is not None:
            trace_fields["receiver_gpu_wait_copy_ms"] = round(receiver_gpu_ms, 6)
            trace_fields["host_minus_receiver_gpu_ms"] = round(
                host_wait_ms - receiver_gpu_ms, 6
            )
        if self._emit_trace:
            _comm_trace("cuda_ipc_get_wait_copy", **trace_fields)

    def _release_references(self) -> None:
        self._completed = True
        self._pool_tensor = None
        self._held_references = ()
        self._start_event = None
        self._done_event = None


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
        self._kv_pools: dict[str, KVPool] = {}
        self._kv_pool_registration_ids: dict[str, str] = {}
        self._kv_pool_storage_handles: dict[
            tuple[str, str], tuple[dict[str, Any], ...]
        ] = {}
        self._remote_kv_pools: dict[tuple[str, str], tuple[torch.Tensor, ...]] = {}
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
            if not _ensure_peer_access(src_index, dst_index):
                raise RuntimeError(
                    "cuda_ipc cross-GPU transfer requires peer access, but "
                    f"GPU {dst_index} cannot access GPU {src_index}; use the "
                    "shm relay for host-staged transfer"
                )
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

    def register_kv_pool(self, pool: KVPool) -> None:
        expected_device = torch.device(self.device)
        if pool.device != expected_device:
            raise ValueError(
                f"KV pool {pool.pool_id!r} is on {pool.device}, but relay uses "
                f"{expected_device}"
            )
        for buffer in pool.buffers:
            if buffer.bytes_per_page % 8 != 0:
                raise ValueError(
                    f"CUDA IPC KV buffer {buffer.name!r} bytes_per_page must be "
                    f"a multiple of 8, got {buffer.bytes_per_page}"
                )
        self._kv_pools[pool.pool_id] = pool
        self._kv_pool_registration_ids.setdefault(
            pool.pool_id,
            f"{self.engine_id}:{os.getpid()}:{uuid.uuid4().hex}",
        )

    def _export_kv_source_pool(
        self,
        pool_id: str,
        *,
        destination_registration_id: str,
    ) -> dict[str, Any]:
        pool = self._kv_pools.get(pool_id)
        if pool is None:
            raise KeyError(f"unknown cuda_ipc KV pool {pool_id!r}")
        cache_key = (pool_id, destination_registration_id)
        storage_handles = self._kv_pool_storage_handles.get(cache_key)
        if storage_handles is None:
            storage_handles = tuple(
                _dump_cuda_storage_handle(buffer.byte_view()) for buffer in pool.buffers
            )
            self._kv_pool_storage_handles[cache_key] = storage_handles
        return {
            "engine_id": self.engine_id,
            "cuda_ipc_kv": {
                "registration_id": self._kv_pool_registration_ids[pool_id],
                "device_id": self.device_id,
                "storages": list(storage_handles),
            },
        }

    def prepare_kv_destination(self, pool_id: str) -> dict[str, Any]:
        pool = self._kv_pools.get(pool_id)
        if pool is None:
            raise KeyError(f"unknown cuda_ipc KV pool {pool_id!r}")
        return {
            "cuda_ipc_kv_destination": {
                "registration_id": self._kv_pool_registration_ids[pool_id],
            },
        }

    async def put_kv_pages(
        self,
        *,
        source_pool_id: str,
        source_page_indices: tuple[int, ...],
        destination_ref: dict[str, Any],
        transfer_id: str | None = None,
    ) -> _ReceiverAckOperation:
        pool = self._kv_pools.get(source_pool_id)
        if pool is None:
            raise KeyError(f"unknown cuda_ipc KV pool {source_pool_id!r}")
        destination_registration_id = destination_ref["cuda_ipc_kv_destination"][
            "registration_id"
        ]

        device = pool.device
        stream = torch.cuda.current_stream(device)
        ready_event = torch.cuda.Event(interprocess=True)
        with torch.cuda.device(device), torch.cuda.stream(stream):
            ready_event.record(stream)
        relay_info = self._export_kv_source_pool(
            source_pool_id,
            destination_registration_id=destination_registration_id,
        )
        relay_info["transfer_info"] = {
            "size": sum(
                buffer.bytes_per_page * len(source_page_indices)
                for buffer in pool.buffers
            )
        }
        relay_info["cuda_ipc_kv"] = dict(relay_info["cuda_ipc_kv"])
        relay_info["cuda_ipc_kv"]["ready_event"] = ready_event.ipc_handle()
        _comm_trace(
            "cuda_ipc_kv_put",
            transfer_id=transfer_id,
            source_pool_id=source_pool_id,
            num_pages=len(source_page_indices),
            bytes=relay_info["transfer_info"]["size"],
            device_id=device.index if device.index is not None else 0,
        )
        return _ReceiverAckOperation(
            relay_info,
            held_references=(
                ready_event,
                tuple(buffer.byte_view() for buffer in pool.buffers),
            ),
        )

    async def get_kv_pages(
        self,
        metadata: dict[str, Any],
        *,
        destination_pool_id: str,
        source_page_indices: tuple[int, ...],
        destination_page_indices: tuple[int, ...],
        request_id: str,
        transfer_id: str | None = None,
    ) -> CudaIpcGetOperation:
        destination = self._kv_pools.get(destination_pool_id)
        if destination is None:
            raise KeyError(f"unknown cuda_ipc KV pool {destination_pool_id!r}")
        source_meta = metadata["cuda_ipc_kv"]

        destination_device = destination.device
        source_device_id = int(source_meta["device_id"])
        destination_device_id = int(destination_device.index or 0)
        if 0 <= source_device_id < torch.cuda.device_count():
            if (
                source_device_id != destination_device_id
                and not torch.cuda.can_device_access_peer(
                    destination_device_id,
                    source_device_id,
                )
            ):
                _comm_trace(
                    "cuda_ipc_kv_peer_access_denied",
                    transfer_id=transfer_id,
                    request_id=request_id,
                    source_device_id=source_device_id,
                    destination_device_id=destination_device_id,
                )
                raise RuntimeError(
                    "direct CUDA-IPC KV transfer requires GPU peer access; "
                    f"cuda:{destination_device_id} cannot access "
                    f"cuda:{source_device_id}"
                )
            _ensure_peer_access(source_device_id, destination_device_id)
        source_engine_id = metadata["engine_id"]
        source_registration_id = source_meta["registration_id"]
        remote_pool_key = (source_engine_id, source_registration_id)
        source_buffers = self._remote_kv_pools.get(remote_pool_key)
        remote_pool_cached = source_buffers is not None
        if source_buffers is None:
            source_buffers = tuple(
                _load_cuda_storage_handle(storage, device=destination_device)
                for storage in source_meta["storages"]
            )
            self._remote_kv_pools[remote_pool_key] = source_buffers

        copy_args = tuple(
            (source, target.byte_view(), target.bytes_per_page)
            for source, target in zip(
                source_buffers,
                destination.buffers,
                strict=True,
            )
        )

        ready_event = torch.cuda.Event.from_ipc_handle(
            destination_device,
            source_meta["ready_event"],
        )
        source_indices = torch.tensor(
            source_page_indices,
            dtype=torch.int64,
            device=destination_device,
        )
        destination_indices = torch.tensor(
            destination_page_indices,
            dtype=torch.int64,
            device=destination_device,
        )
        stream = torch.cuda.current_stream(destination_device)
        done_event = torch.cuda.Event()
        from sgl_kernel import transfer_kv_per_layer_mla

        with torch.cuda.device(destination_device), torch.cuda.stream(stream):
            stream.wait_event(ready_event)
            try:
                for source, target, item_size in copy_args:
                    transfer_kv_per_layer_mla(
                        src=source,
                        dst=target,
                        src_indices=source_indices,
                        dst_indices=destination_indices,
                        item_size=item_size,
                    )
                done_event.record(stream)
            except Exception:
                with suppress(Exception):
                    stream.synchronize()
                raise
        transfer_size = sum(
            buffer.bytes_per_page * len(source_page_indices)
            for buffer in destination.buffers
        )
        _comm_trace(
            "cuda_ipc_kv_get",
            transfer_id=transfer_id,
            request_id=request_id,
            destination_pool_id=destination_pool_id,
            num_pages=len(source_page_indices),
            bytes=transfer_size,
            source_device_id=source_device_id,
            destination_device_id=destination_device_id,
            cross_device=source_device_id != destination_device_id,
            remote_pool_cached=remote_pool_cached,
        )
        return CudaIpcGetOperation(
            done_event,
            None,
            -1,
            0,
            request_id,
            transfer_size,
            destination_device_id,
            self._wait_executor,
            held_references=(
                ready_event,
                source_buffers,
                source_indices,
                destination_indices,
            ),
            finish_on_interrupt=True,
            emit_trace=False,
        )

    def cleanup(self, request_id: str) -> None:
        pass

    def close(self) -> None:
        self._remote_pools.clear()
        self._remote_kv_pools.clear()
        self._kv_pool_storage_handles.clear()
        self._kv_pool_registration_ids.clear()
        self._kv_pools.clear()
        self._pool_storage_handles.clear()
        self._pool_tensor = None
        self._allocator = None
        self._wait_executor.shutdown(wait=False, cancel_futures=True)
