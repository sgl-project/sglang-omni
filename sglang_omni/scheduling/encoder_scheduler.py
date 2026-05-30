# SPDX-License-Identifier: Apache-2.0
"""TP-aware scheduler for SGLang-backed encoder stages.

Same public shape as :class:`SimpleScheduler` (``inbox`` / ``outbox`` /
``start`` / ``stop`` / ``abort``) so :class:`Stage` does not branch on
scheduler type. Adds an explicit two-channel broadcast in
``_recv_messages``:

1. **Metadata** over the SGLang TP CPU group via ``broadcast_pyobj`` —
   pickles only the dict skeleton, never the tensor payload bytes.
2. **Tensors** over the SGLang TP device group via ``dist.broadcast``
   on cuda:0.

A small ``all_gather_object`` "alloc-ok?" handshake between the two
broadcasts prevents a non-entry-rank OOM mid-allocation from leaving
the entry rank stuck in ``dist.broadcast``.

The scheduler is correct in both lanes:

- ``tp_size == 1``: skip the broadcasts; still strip-and-lift CPU shm
  tensors to ``cuda:0`` because the upstream ``get_image_feature`` /
  ``get_video_feature`` call ``.type(dtype)`` only — they do not move
  tensors to the model's device.
- ``tp_size  > 1``: drain the inbox on the entry rank only; broadcast
  inputs to non-entry ranks; run the same ``build_batch`` /
  ``encode_batch`` / ``slice_results`` pipeline on every rank;
  emit results from the entry rank only.

Naming note: this module uses ``entry_rank`` / ``non-entry rank`` to
describe the rank-0-vs-rest asymmetry. The asymmetry is just "who
owns external IO" — there's no leader election or failover. The
Stage-level ``single/leader/follower`` role split is a separate
abstraction layer that this scheduler doesn't touch.

See ``docs/developer_reference/encoder_tp_path_b_design.md`` for the
load-bearing design notes.
"""
from __future__ import annotations

import collections
import dataclasses
import json
import logging
import os
import queue as _queue_mod
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.distributed as dist

from sglang.srt.utils import broadcast_pyobj

from sglang_omni.pipeline.relay_io import extract_tensors, restore_tensors
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage
from sglang_omni.utils.gpu_memory import (
    format_bytes_gib,
    get_gpu_startup_lock_path,
    get_process_gpu_memory_bytes,
)

if TYPE_CHECKING:
    from sglang_omni.model_runner.sglang_encoder_runner import SGLangEncoderRunner
    from sglang_omni.models.qwen3_omni.encoder_adapters import (
        BatchPlan,
        EncoderAdapter,
    )

logger = logging.getLogger(__name__)

# Tagged-dict sentinel used to ship recv-time errors over the same
# CPU-group ``broadcast_pyobj`` slot the success path uses. Identity
# sentinels (e.g. ``object()``) do not survive the pickle round-trip
# inside ``broadcast_pyobj``; a kind-string does.
_RECV_ERROR_KIND = "encoder_recv_error"
_GPU_GUARD_TRANSIENT_MARGIN_MIB = 256
_GPU_GUARD_ALLOCATOR_MARGIN_MIB = 256


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_mib(name: str, default_mib: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default_mib) * 1024**2
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; expected integer MiB", name, raw)
        value = int(default_mib)
    return max(value, 0) * 1024**2


def _reservation_path_for_gpu(logical_gpu_id: int) -> Path:
    base_dir = os.getenv("SGLANG_OMNI_ENCODER_GPU_GUARD_DIR")
    startup_lock = get_gpu_startup_lock_path(
        logical_gpu_id,
        base_dir=base_dir or tempfile.gettempdir(),
    )
    suffix = startup_lock.name.removeprefix("sglang_omni_gpu_")
    suffix = suffix.removesuffix("_startup.lock")
    return startup_lock.with_name(
        f"sglang_omni_encoder_gpu_{suffix}_reservations.json"
    )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclasses.dataclass(slots=True)
class _TensorSpec:
    """Lightweight description of a tensor for the metadata broadcast.

    Carries the typed ``torch.dtype`` (not the stringified form
    ``relay_io.extract_tensors`` produces, which would force a parser on
    the non-entry-rank side).
    """
    path: str
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclasses.dataclass(slots=True)
class _RecvPathTiming:
    """Fine-grained timing for one EncoderScheduler recv/fanout pass."""

    inbox_admission_ms: float = 0.0
    strip_h2d_ms: float = 0.0
    metadata_broadcast_ms: float = 0.0
    follower_allocation_ms: float = 0.0
    allocation_handshake_ms: float = 0.0
    tensor_broadcast_ms: float = 0.0
    rank_wait_skew_ms: float = 0.0
    rank_arrival_skew_ms: float = 0.0


@dataclasses.dataclass(slots=True)
class _GpuGuardDecision:
    allowed: bool
    reason: str = ""
    batch_cost_bytes: int = 0
    free_bytes: int | None = None
    total_bytes: int | None = None
    reserved_bytes: int = 0
    transient_margin_bytes: int = 0
    allocator_margin_bytes: int = 0

    @property
    def required_bytes(self) -> int:
        return (
            int(self.batch_cost_bytes)
            + int(self.reserved_bytes)
            + int(self.transient_margin_bytes)
            + int(self.allocator_margin_bytes)
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _GpuReservation:
    key: str
    bytes: int


class _EncoderGpuMemoryGuard:
    """Best-effort whole-GPU admission guard shared by colocated encoders.

    The activation cost estimator is still the source of the projected
    encoder bytes. This guard adds current whole-GPU free memory and a
    cross-process reservation file so image/audio encoder processes on the
    same physical GPU do not all admit work against the same free bytes.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        logical_gpu_id: int,
        stage_name: str,
        transient_margin_bytes: int,
        allocator_margin_bytes: int,
        reservation_path: Path,
    ) -> None:
        self.enabled = enabled
        self.logical_gpu_id = int(logical_gpu_id)
        self.stage_name = stage_name
        self.transient_margin_bytes = max(int(transient_margin_bytes), 0)
        self.allocator_margin_bytes = max(int(allocator_margin_bytes), 0)
        self.reservation_path = reservation_path
        self._pid = os.getpid()

    @classmethod
    def from_runner(
        cls,
        runner: Any,
        *,
        stage_name: str,
    ) -> "_EncoderGpuMemoryGuard":
        enabled = _env_flag("SGLANG_OMNI_ENCODER_GPU_GUARD", default=True)
        device = getattr(runner, "device", None)
        if not isinstance(device, torch.device):
            try:
                device = torch.device(device)
            except Exception:
                device = None
        if device is None or device.type != "cuda":
            enabled = False
            logical_gpu_id = 0
        else:
            logical_gpu_id = int(device.index or 0)
            if not torch.cuda.is_available():
                enabled = False

        reservation_path = _reservation_path_for_gpu(logical_gpu_id)
        return cls(
            enabled=enabled,
            logical_gpu_id=logical_gpu_id,
            stage_name=stage_name,
            transient_margin_bytes=_env_mib(
                "SGLANG_OMNI_ENCODER_GPU_GUARD_TRANSIENT_MARGIN_MIB",
                _GPU_GUARD_TRANSIENT_MARGIN_MIB,
            ),
            allocator_margin_bytes=_env_mib(
                "SGLANG_OMNI_ENCODER_GPU_GUARD_ALLOCATOR_MARGIN_MIB",
                _GPU_GUARD_ALLOCATOR_MARGIN_MIB,
            ),
            reservation_path=reservation_path,
        )

    def check(self, batch_cost_bytes: int) -> _GpuGuardDecision:
        if not self.enabled:
            return _GpuGuardDecision(
                allowed=True,
                reason="disabled",
                batch_cost_bytes=max(int(batch_cost_bytes), 0),
            )
        return self._with_state(batch_cost_bytes, reserve=False)[0]

    def reserve(
        self,
        batch_cost_bytes: int,
    ) -> tuple[_GpuGuardDecision, _GpuReservation | None]:
        if not self.enabled:
            return (
                _GpuGuardDecision(
                    allowed=True,
                    reason="disabled",
                    batch_cost_bytes=max(int(batch_cost_bytes), 0),
                ),
                None,
            )
        return self._with_state(batch_cost_bytes, reserve=True)

    def release(self, reservation: _GpuReservation | None) -> None:
        if reservation is None or not self.enabled:
            return
        self.reservation_path.parent.mkdir(parents=True, exist_ok=True)
        import fcntl

        with open(self.reservation_path, "a+", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            state = self._read_state(f)
            reservations = state.setdefault("reservations", {})
            reservations.pop(reservation.key, None)
            self._write_state(f, state)

    def _with_state(
        self,
        batch_cost_bytes: int,
        *,
        reserve: bool,
    ) -> tuple[_GpuGuardDecision, _GpuReservation | None]:
        batch_cost = max(int(batch_cost_bytes), 0)
        self.reservation_path.parent.mkdir(parents=True, exist_ok=True)
        import fcntl

        with open(self.reservation_path, "a+", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            state = self._read_state(f)
            reservations = self._prune_reservations(
                state.setdefault("reservations", {})
            )
            free_bytes, total_bytes = self._mem_get_info()
            if free_bytes is None or total_bytes is None:
                self._write_state(f, state)
                return (
                    _GpuGuardDecision(
                        allowed=True,
                        reason="telemetry_unavailable",
                        batch_cost_bytes=batch_cost,
                        reserved_bytes=sum(
                            int(item.get("bytes", 0))
                            for item in reservations.values()
                        ),
                        transient_margin_bytes=self.transient_margin_bytes,
                        allocator_margin_bytes=self.allocator_margin_bytes,
                    ),
                    None,
                )

            reserved_bytes = sum(
                int(item.get("bytes", 0)) for item in reservations.values()
            )
            decision = _GpuGuardDecision(
                allowed=True,
                reason="fits",
                batch_cost_bytes=batch_cost,
                free_bytes=int(free_bytes),
                total_bytes=int(total_bytes),
                reserved_bytes=reserved_bytes,
                transient_margin_bytes=self.transient_margin_bytes,
                allocator_margin_bytes=self.allocator_margin_bytes,
            )
            if decision.required_bytes > int(free_bytes):
                decision.allowed = False
                decision.reason = "insufficient_free_memory"
                self._write_state(f, state)
                return decision, None

            token = None
            if reserve and batch_cost > 0:
                key = f"{self._pid}:{self.stage_name}:{uuid.uuid4().hex}"
                reservations[key] = {
                    "pid": self._pid,
                    "stage": self.stage_name,
                    "bytes": batch_cost,
                    "created_at": time.time(),
                }
                token = _GpuReservation(key=key, bytes=batch_cost)
            self._write_state(f, state)
            return decision, token

    def _mem_get_info(self) -> tuple[int | None, int | None]:
        try:
            with torch.cuda.device(self.logical_gpu_id):
                free_bytes, total_bytes = torch.cuda.mem_get_info()
            return int(free_bytes), int(total_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.debug("encoder GPU guard telemetry unavailable: %s", exc)
            return None, None

    def _read_state(self, f: Any) -> dict[str, Any]:
        f.seek(0)
        raw = f.read()
        if not raw.strip():
            return {"reservations": {}}
        try:
            state = json.loads(raw)
        except json.JSONDecodeError:
            return {"reservations": {}}
        if not isinstance(state, dict):
            return {"reservations": {}}
        if not isinstance(state.get("reservations"), dict):
            state["reservations"] = {}
        return state

    def _write_state(self, f: Any, state: dict[str, Any]) -> None:
        f.seek(0)
        f.truncate()
        json.dump(state, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())

    def _prune_reservations(
        self,
        reservations: dict[str, Any],
    ) -> dict[str, Any]:
        stale: list[str] = []
        for key, item in reservations.items():
            pid = int(item.get("pid", -1)) if isinstance(item, dict) else -1
            if pid <= 0 or not _pid_alive(pid):
                stale.append(key)
        for key in stale:
            reservations.pop(key, None)
        return reservations


class BatchCollectError(RuntimeError):
    """Raised when batch admission fails after draining one or more messages.

    Carries the drained ``messages`` so the scheduler can emit one
    request-level error per request instead of silently dropping them.
    """

    def __init__(
        self,
        messages: list[IncomingMessage],
        error: BaseException,
    ) -> None:
        super().__init__(str(error))
        self.messages = messages
        self.error = error


class EncoderScheduler:
    """Inbox -> two-channel broadcast -> encoder forward -> outbox.

    Public contract identical to :class:`SimpleScheduler`. ``Stage`` is
    unaware of the TP shape — it just hands inputs to ``inbox`` and
    drains ``outbox``.
    """

    def __init__(
        self,
        runner: "SGLangEncoderRunner",
        adapter: "EncoderAdapter",
        *,
        max_batch_size: int = 32,
        max_batch_wait_ms: int = 50,
        request_cost_fn: Callable[[Any], int] | None = None,
        batch_cost_fn: Callable[[list[Any]], int] | None = None,
        max_batch_cost: int | None = None,
        max_single_request_cost: int | None = None,
        gpu_memory_guard: _EncoderGpuMemoryGuard | None = None,
        output_cache: Any | None = None,
    ) -> None:
        self.runner = runner
        self.adapter = adapter
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self._max_batch_size = max(int(max_batch_size), 1)
        self._max_batch_wait_s = max(float(max_batch_wait_ms), 0.0) / 1000.0
        self._request_cost_fn = request_cost_fn
        self._batch_cost_fn = batch_cost_fn
        self._max_batch_cost = (
            max(int(max_batch_cost), 0) if max_batch_cost is not None else None
        )
        self._max_single_request_cost = (
            max(int(max_single_request_cost), 0)
            if max_single_request_cost is not None
            else None
        )
        self._output_cache = output_cache
        self._pending_messages: collections.deque[IncomingMessage] = (
            collections.deque()
        )
        self._gpu_memory_guard = (
            gpu_memory_guard
            if gpu_memory_guard is not None
            else _EncoderGpuMemoryGuard.from_runner(
                runner,
                stage_name=self._stage_name,
            )
        )
        self._active_reservation: _GpuReservation | None = None
        self._running = False
        self._aborted_request_ids: set[str] = set()
        self._last_recv_timing = _RecvPathTiming()
        self._sync_recv_timing = os.getenv(
            "SGLANG_OMNI_ENCODER_TIMING_DETAIL",
            "",
        ).lower() in {"1", "true", "yes", "on"}
        self._memory_detail = _env_flag(
            "SGLANG_OMNI_ENCODER_MEMORY_DETAIL",
            default=False,
        )

    @property
    def _stage_name(self) -> str:
        return str(getattr(self.adapter, "stage_name", "encoder"))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Run the scheduler loop on the current thread.

        Three error domains, not one catch-all:

        1. **Recoverable pre-forward** (``_recv_messages``,
           ``build_batch``): both synchronize through the TP CPU group
           via :meth:`_gather_pre_forward_error` *before* any model
           collective starts. On any rank failure, the entry rank emits
           one ``OutgoingMessage(type="error")`` per drained request and
           every rank ``continue``s into the next loop iteration.

        2. **Fatal forward** (``encode_batch``): once we've entered
           upstream SGLang TP collectives (``ColumnParallelLinear``,
           ``RowParallelLinear``, NCCL), one rank cannot safely recover
           with a CPU gather — peers may still be blocked in NCCL. The
           rank that observes the exception calls
           :meth:`_fatal_tp_forward_error`, which exits the process
           non-zero so ``StageGroup`` / ``MultiProcessPipelineRunner``
           tears down the whole TP group and the runner's monitor fails
           outstanding Coordinator futures.

        3. **Recoverable post-forward** (``slice_results``): runs only
           on the entry rank after ``encode_batch`` returned on every
           rank. It can emit per-request errors locally and continue.
        """
        self._running = True
        while self._running:
            iter_started = time.perf_counter()
            self._reset_memory_peak_stats()
            recv_started = time.perf_counter()
            messages, recv_err = self._recv_messages()
            recv_ms = (time.perf_counter() - recv_started) * 1000.0
            if self._gather_pre_forward_error(recv_err):
                if self.runner.is_entry_rank:
                    self._emit_error(
                        messages,
                        recv_err
                        if recv_err is not None
                        else RuntimeError("peer-rank encoder recv failed"),
                    )
                self._release_active_reservation()
                continue
            if not messages:
                self._release_active_reservation()
                continue

            plan = None
            build_err: BaseException | None = None
            build_started = time.perf_counter()
            try:
                plan = self.adapter.build_batch(messages)
            except Exception as exc:  # noqa: BLE001
                build_err = exc
            build_ms = (time.perf_counter() - build_started) * 1000.0

            if self._gather_pre_forward_error(build_err):
                if self.runner.is_entry_rank:
                    self._emit_error(
                        messages,
                        build_err
                        if build_err is not None
                        else RuntimeError("peer-rank encoder build_batch failed"),
                    )
                self._release_active_reservation()
                continue

            self._log_memory_mark(
                "after_build_before_forward",
                batch_size=len(messages),
            )
            forward_started = time.perf_counter()
            try:
                raw = self.runner.encode_batch(plan)
            except Exception as exc:  # noqa: BLE001
                # Production: _fatal_tp_forward_error calls os._exit(1)
                # and never returns; the runner's _monitor_children
                # then fails outstanding Coordinator futures.
                # Tests: monkey-patched fatal handler returns. In that
                # case we exit the loop cleanly without re-raising —
                # re-raising would crash the scheduler thread, which
                # under Stage._handle_scheduler_crash would tear down
                # the stage even in tests where we want to assert state
                # post-fault.
                self._fatal_tp_forward_error(exc)
                self._running = False
                self._release_active_reservation()
                return
            forward_ms = (time.perf_counter() - forward_started) * 1000.0
            self._log_memory_mark("after_forward", batch_size=len(messages))

            if not self.runner.is_entry_rank:
                logger.info(
                    "encoder_batch_timing stage=%s tp_rank=%d/%d "
                    "batch_size=%d recv_ms=%.3f "
                    "inbox_admission_ms=%.3f strip_h2d_ms=%.3f "
                    "metadata_broadcast_ms=%.3f "
                    "follower_allocation_ms=%.3f "
                    "allocation_handshake_ms=%.3f "
                    "tensor_broadcast_ms=%.3f rank_wait_skew_ms=%.3f "
                    "rank_arrival_skew_ms=%.3f build_ms=%.3f "
                    "forward_ms=%.3f slice_ms=0.000 total_ms=%.3f",
                    self._stage_name,
                    self.runner.tp_rank,
                    self.runner.tp_size,
                    len(messages),
                    recv_ms,
                    self._last_recv_timing.inbox_admission_ms,
                    self._last_recv_timing.strip_h2d_ms,
                    self._last_recv_timing.metadata_broadcast_ms,
                    self._last_recv_timing.follower_allocation_ms,
                    self._last_recv_timing.allocation_handshake_ms,
                    self._last_recv_timing.tensor_broadcast_ms,
                    self._last_recv_timing.rank_wait_skew_ms,
                    self._last_recv_timing.rank_arrival_skew_ms,
                    build_ms,
                    forward_ms,
                    (time.perf_counter() - iter_started) * 1000.0,
                )
                self._release_active_reservation()
                self._log_memory_mark(
                    "after_cleanup_synchronize",
                    batch_size=len(messages),
                    synchronize=True,
                )
                continue

            slice_started = time.perf_counter()
            try:
                results = self.adapter.slice_results(raw, plan, messages)
            except Exception as exc:  # noqa: BLE001
                logger.exception("EncoderScheduler slice_results failed")
                self._emit_error(messages, exc)
                self._release_active_reservation()
                continue
            slice_ms = (time.perf_counter() - slice_started) * 1000.0
            self._store_cached_outputs(messages, results)

            for msg, out in zip(messages, results):
                if msg.request_id in self._aborted_request_ids:
                    continue
                self.outbox.put(
                    OutgoingMessage(
                        request_id=msg.request_id,
                        type="result",
                        data=out,
                    )
                )
            self._log_memory_mark(
                "after_slice_output_staging",
                batch_size=len(messages),
            )
            logger.info(
                "encoder_batch_timing stage=%s tp_rank=%d/%d "
                "batch_size=%d recv_ms=%.3f "
                "inbox_admission_ms=%.3f strip_h2d_ms=%.3f "
                "metadata_broadcast_ms=%.3f follower_allocation_ms=%.3f "
                "allocation_handshake_ms=%.3f tensor_broadcast_ms=%.3f "
                "rank_wait_skew_ms=%.3f rank_arrival_skew_ms=%.3f "
                "build_ms=%.3f forward_ms=%.3f slice_ms=%.3f total_ms=%.3f",
                self._stage_name,
                self.runner.tp_rank,
                self.runner.tp_size,
                len(messages),
                recv_ms,
                self._last_recv_timing.inbox_admission_ms,
                self._last_recv_timing.strip_h2d_ms,
                self._last_recv_timing.metadata_broadcast_ms,
                self._last_recv_timing.follower_allocation_ms,
                self._last_recv_timing.allocation_handshake_ms,
                self._last_recv_timing.tensor_broadcast_ms,
                self._last_recv_timing.rank_wait_skew_ms,
                self._last_recv_timing.rank_arrival_skew_ms,
                build_ms,
                forward_ms,
                slice_ms,
                (time.perf_counter() - iter_started) * 1000.0,
            )
            self._release_active_reservation()
            self._log_memory_mark(
                "after_cleanup_synchronize",
                batch_size=len(messages),
                synchronize=True,
            )

    def _reset_memory_peak_stats(self) -> None:
        if not self._memory_detail:
            return
        device = getattr(self.runner, "device", None)
        if not isinstance(device, torch.device) or device.type != "cuda":
            return
        try:
            with torch.cuda.device(device):
                torch.cuda.reset_peak_memory_stats(device)
        except Exception as exc:  # noqa: BLE001
            logger.debug("encoder memory peak reset failed: %s", exc)

    def _log_memory_mark(
        self,
        mark: str,
        *,
        batch_size: int = 0,
        synchronize: bool = False,
    ) -> None:
        if not self._memory_detail:
            return
        if batch_size <= 0:
            return
        device = getattr(self.runner, "device", None)
        if not isinstance(device, torch.device) or device.type != "cuda":
            return
        try:
            with torch.cuda.device(device):
                if synchronize:
                    torch.cuda.synchronize(device)
                allocated = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                max_allocated = torch.cuda.max_memory_allocated(device)
                max_reserved = torch.cuda.max_memory_reserved(device)
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            nvml_process_bytes = get_process_gpu_memory_bytes(int(device.index or 0))
        except Exception as exc:  # noqa: BLE001
            logger.debug("encoder memory mark failed mark=%s: %s", mark, exc)
            return
        logger.info(
            "encoder_memory_mark stage=%s tp_rank=%d/%d mark=%s "
            "batch_size=%d allocated_bytes=%d reserved_bytes=%d "
            "max_allocated_bytes=%d max_reserved_bytes=%d "
            "free_bytes=%d total_bytes=%d nvml_process_bytes=%s",
            self._stage_name,
            self.runner.tp_rank,
            self.runner.tp_size,
            mark,
            batch_size,
            allocated,
            reserved,
            max_allocated,
            max_reserved,
            int(free_bytes),
            int(total_bytes),
            "None" if nvml_process_bytes is None else str(nvml_process_bytes),
        )

    def _gather_pre_forward_error(
        self,
        local_err: BaseException | None,
    ) -> bool:
        """Synchronize recoverable recv/build errors before model collectives.

        Returns True iff *any* rank reported a failure. The TP CPU group
        gather marshals only a picklable boolean — the exception object
        stays local, so each rank emits its own request-level error.
        """
        if self.runner.tp_size <= 1:
            return local_err is not None
        err_flags: list[bool] = [False] * self.runner.tp_size
        dist.all_gather_object(
            err_flags,
            local_err is not None,
            group=self.runner.tp_group.cpu_group,
        )
        return any(err_flags)

    def _fatal_tp_forward_error(self, error: BaseException) -> None:
        """Exit non-zero after a TP forward fault.

        Once ``encode_batch`` has entered upstream SGLang TP collectives,
        a rank-local exception cannot be recovered through a post-hoc CPU
        gather: peers may still be blocked in NCCL and never reach the
        gather. Force a child-process failure so
        :class:`MultiProcessPipelineRunner` tears down the whole TP group
        and fails outstanding Coordinator futures from the parent side.

        Overridable by tests via monkey-patch (the test stub records the
        error and *does* return so the test runner can assert on it).
        """
        logger.exception(
            "Fatal TP encoder forward failure on rank %d/%d: %r",
            self.runner.tp_rank, self.runner.tp_size, error,
        )
        os._exit(1)

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        """Mark a request as aborted; results emitted later will be dropped."""
        self._aborted_request_ids.add(request_id)
        # bound the set so it cannot grow forever in long-running servers
        if len(self._aborted_request_ids) > 10000:
            keep = list(self._aborted_request_ids)[-5000:]
            self._aborted_request_ids = set(keep)

    # ------------------------------------------------------------------
    # Recv path: inbox drain + two-channel broadcast
    # ------------------------------------------------------------------

    def _next_message(self, *, block: bool = True) -> IncomingMessage | None:
        if self._pending_messages:
            return self._pending_messages.popleft()
        try:
            if block:
                return self.inbox.get(timeout=0.1)
            return self.inbox.get_nowait()
        except _queue_mod.Empty:
            return None

    def _cached_output(self, msg: IncomingMessage) -> Any | None:
        if (
            self._output_cache is None
            or msg.type != "new_request"
            or not self.runner.is_entry_rank
        ):
            return None
        lookup = getattr(self.adapter, "lookup_cached_output", None)
        if lookup is None:
            return None
        return lookup(msg, self._output_cache)

    def _emit_cached_output_if_available(self, msg: IncomingMessage) -> bool:
        cached = self._cached_output(msg)
        if cached is None:
            return False
        if msg.request_id not in self._aborted_request_ids:
            self.outbox.put(
                OutgoingMessage(
                    request_id=msg.request_id,
                    type="result",
                    data=cached,
                )
            )
        return True

    def _store_cached_outputs(
        self,
        messages: list[IncomingMessage],
        outputs: list[Any],
    ) -> None:
        if self._output_cache is None or not self.runner.is_entry_rank:
            return
        store = getattr(self.adapter, "store_cached_output", None)
        if store is None:
            return
        for msg, out in zip(messages, outputs):
            store(msg, out, self._output_cache)

    def _message_cost(self, msg: IncomingMessage) -> int:
        if self._request_cost_fn is None or msg.type != "new_request":
            return 0
        return max(int(self._request_cost_fn(msg.data)), 0)

    def _batch_cost(self, msgs: list[IncomingMessage]) -> int:
        if self._batch_cost_fn is not None:
            return max(int(self._batch_cost_fn([msg.data for msg in msgs])), 0)
        return sum(self._message_cost(msg) for msg in msgs)

    def _check_gpu_guard(self, batch_cost: int) -> _GpuGuardDecision:
        return self._gpu_memory_guard.check(batch_cost)

    def _reserve_gpu_guard(
        self,
        batch_cost: int,
    ) -> tuple[_GpuGuardDecision, _GpuReservation | None]:
        return self._gpu_memory_guard.reserve(batch_cost)

    def _release_active_reservation(self) -> None:
        reservation = self._active_reservation
        if reservation is None:
            return
        self._active_reservation = None
        self._gpu_memory_guard.release(reservation)
        logger.info(
            "encoder_reservation stage=%s decision=release bytes=%d",
            self._stage_name,
            reservation.bytes,
        )

    def _format_gpu_guard_error(self, decision: _GpuGuardDecision) -> RuntimeError:
        return RuntimeError(
            "encoder whole-GPU guard rejected admission: "
            f"reason={decision.reason} "
            f"batch_cost={format_bytes_gib(decision.batch_cost_bytes)} "
            f"reserved={format_bytes_gib(decision.reserved_bytes)} "
            f"transient_margin={format_bytes_gib(decision.transient_margin_bytes)} "
            f"allocator_margin={format_bytes_gib(decision.allocator_margin_bytes)} "
            f"required={format_bytes_gib(decision.required_bytes)} "
            f"free={format_bytes_gib(decision.free_bytes)} "
            f"total={format_bytes_gib(decision.total_bytes)}"
        )

    def _reserve_admitted_batch(
        self,
        batch: list[IncomingMessage],
        batch_cost: int,
    ) -> tuple[list[IncomingMessage], int]:
        while batch:
            decision, reservation = self._reserve_gpu_guard(batch_cost)
            if decision.allowed:
                self._active_reservation = reservation
                if reservation is not None:
                    logger.info(
                        "encoder_reservation stage=%s decision=reserve "
                        "batch_size=%d batch_cost=%d reserved_bytes=%d "
                        "free_bytes=%s required_bytes=%d",
                        self._stage_name,
                        len(batch),
                        batch_cost,
                        reservation.bytes,
                        decision.free_bytes,
                        decision.required_bytes,
                    )
                return batch, batch_cost

            if len(batch) == 1:
                if decision.reserved_bytes > 0:
                    self._pending_messages.appendleft(batch[0])
                    logger.info(
                        "encoder_admission stage=%s decision=defer reason=gpu_guard "
                        "batch_size=1 batch_cost=%d free_bytes=%s "
                        "reserved_bytes=%d required_bytes=%d",
                        self._stage_name,
                        batch_cost,
                        decision.free_bytes,
                        decision.reserved_bytes,
                        decision.required_bytes,
                    )
                    time.sleep(min(max(self._max_batch_wait_s, 0.001), 0.01))
                    return [], 0
                logger.info(
                    "encoder_admission stage=%s decision=reject reason=gpu_guard "
                    "batch_size=1 batch_cost=%d free_bytes=%s "
                    "reserved_bytes=%d required_bytes=%d",
                    self._stage_name,
                    batch_cost,
                    decision.free_bytes,
                    decision.reserved_bytes,
                    decision.required_bytes,
                )
                raise BatchCollectError(batch, self._format_gpu_guard_error(decision))

            deferred = batch.pop()
            self._pending_messages.appendleft(deferred)
            batch_cost = self._batch_cost(batch)
            logger.info(
                "encoder_admission stage=%s decision=shrink reason=gpu_guard "
                "deferred_request_id=%s batch_size=%d batch_cost=%d "
                "free_bytes=%s reserved_bytes=%d required_bytes=%d",
                self._stage_name,
                deferred.request_id,
                len(batch),
                batch_cost,
                decision.free_bytes,
                decision.reserved_bytes,
                decision.required_bytes,
            )

        return [], 0

    def _validate_single_message(self, msg: IncomingMessage) -> None:
        if self._max_single_request_cost is None or msg.type != "new_request":
            return
        cost = self._message_cost(msg)
        if cost > self._max_single_request_cost:
            raise ValueError(
                f"encoder request cost {cost} exceeds "
                f"max_single_request_cost={self._max_single_request_cost}"
            )

    def _collect_batch_from_inbox(self) -> list[IncomingMessage]:
        """Drain the inbox into an activation-budget-bounded batch.

        The batching loop intentionally mirrors
        :func:`SimpleScheduler._collect_batch`: only ``max_batch_size``,
        ``max_batch_wait_ms`` and ``max_batch_cost`` decide whether the next
        inbox item joins this batch. The whole-GPU guard remains a separate
        final reservation step so runtime headroom checks do not obscure the
        core inbox batching rule.

        Raises:
            BatchCollectError: ``request_cost_fn`` (adapter / model code)
                raised. Carries the drained list so the caller can emit
                one error per request instead of silently dropping them.
        """
        first = self._next_message()
        while first is not None and self._emit_cached_output_if_available(first):
            first = self._next_message(block=False)
        if first is None:
            return []

        if first.type != "new_request":
            return [first]

        batch: list[IncomingMessage] = [first]
        try:
            self._validate_single_message(first)
            batch_cost = self._batch_cost(batch)
        except Exception as exc:  # noqa: BLE001
            raise BatchCollectError(batch, exc) from exc

        deadline = time.monotonic() + self._max_batch_wait_s
        while len(batch) < self._max_batch_size:
            try:
                msg = self.inbox.get_nowait()
            except _queue_mod.Empty:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    msg = self.inbox.get(timeout=remaining)
                except _queue_mod.Empty:
                    break

            if msg.type != "new_request":
                self._pending_messages.append(msg)
                continue
            if self._emit_cached_output_if_available(msg):
                continue

            try:
                self._validate_single_message(msg)
            except Exception as exc:  # noqa: BLE001
                batch.append(msg)  # so the failed request gets an error
                raise BatchCollectError(batch, exc) from exc

            candidate = batch + [msg]
            try:
                candidate_cost = self._batch_cost(candidate)
            except Exception as exc:  # noqa: BLE001
                batch.append(msg)  # so the failed request gets an error
                raise BatchCollectError(batch, exc) from exc
            if self._max_batch_cost is not None:
                if candidate_cost > self._max_batch_cost:
                    logger.info(
                        "encoder_admission stage=%s decision=defer "
                        "reason=activation_budget "
                        "batch_size=%d candidate_batch_size=%d "
                        "batch_cost=%d candidate_cost=%d max_batch_cost=%d",
                        self._stage_name,
                        len(batch),
                        len(candidate),
                        batch_cost,
                        candidate_cost,
                        self._max_batch_cost,
                    )
                    self._pending_messages.appendleft(msg)
                    break
            batch_cost = candidate_cost
            batch.append(msg)
        try:
            batch, batch_cost = self._reserve_admitted_batch(batch, batch_cost)
        except BatchCollectError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BatchCollectError(batch, exc) from exc
        if self._max_batch_cost is not None:
            logger.info(
                "encoder_admission stage=%s decision=admit "
                "batch_size=%d batch_cost=%d max_batch_cost=%d",
                self._stage_name,
                len(batch),
                batch_cost,
                self._max_batch_cost,
            )
        return batch

    def _collect_batch_or_error(
        self,
    ) -> tuple[list[IncomingMessage], BaseException | None]:
        try:
            return self._collect_batch_from_inbox(), None
        except BatchCollectError as exc:
            return exc.messages, exc.error
        except Exception as exc:  # noqa: BLE001
            return [], exc

    def _strip_and_lift(
        self,
        messages: list[IncomingMessage],
    ) -> tuple[
        list[IncomingMessage],
        list[list[torch.Tensor]],
        list[list[_TensorSpec]],
    ]:
        """Extract tensors from each message, lift them to the runner device.

        Returns three parallel lists indexed by message:

        - ``meta_msgs``: deep-copied IncomingMessages whose ``data.data``
          dict tree has had its tensors replaced by ``extract_tensors``
          placeholders. These are the only objects that get pickled by
          ``broadcast_pyobj`` — small dict skeletons, never tensor bytes.
        - ``tensor_lists``: per-message lists of GPU-resident tensors,
          ordered so they can be paired one-to-one with ``specs_lists``.
        - ``specs_lists``: per-message lists of :class:`_TensorSpec`
          carrying ``(path, shape, torch.dtype)``. Followers reconstruct
          their receive placeholders from these.

        The CPU-shm relay path delivers tensors on CPU; the H2D copy here
        is the same memcpy the local v1 encoder forward already does
        before its forward call (``image_encoder.py:154``), so it is not
        new work — just earlier. Entry rank callers must reattach
        ``tensor_lists`` with ``_reattach_lifted_tensors`` before model
        execution; keeping ``meta_msgs`` stripped is what prevents the
        CPU-group metadata broadcast from carrying tensor bytes.
        """
        meta_msgs: list[IncomingMessage] = []
        tensor_lists: list[list[torch.Tensor]] = []
        specs_lists: list[list[_TensorSpec]] = []

        for msg in messages:
            payload = msg.data
            if payload is None or not hasattr(payload, "data"):
                meta_msgs.append(msg)
                tensor_lists.append([])
                specs_lists.append([])
                continue

            stripped, tensor_dict = extract_tensors(payload.data)
            tensors: list[torch.Tensor] = []
            specs: list[_TensorSpec] = []
            for path, t in tensor_dict.items():
                if t.device != self.runner.device:
                    t = t.to(self.runner.device, non_blocking=True)
                tensors.append(t)
                specs.append(
                    _TensorSpec(path=path, shape=tuple(t.shape), dtype=t.dtype)
                )

            # Keep the payload stripped for CPU-group metadata broadcast.
            # Entry rank reattaches ``tensors`` after the broadcast path;
            # followers reattach their receive placeholders after device
            # broadcast. Do not restore tensors here, or ``broadcast_pyobj``
            # will pickle the tensor bytes before the device broadcast.
            payload_cls = type(payload)
            new_payload = payload_cls(
                request_id=payload.request_id,
                request=payload.request,
                data=stripped,
            )
            meta_msg = dataclasses.replace(msg, data=new_payload)
            meta_msgs.append(meta_msg)
            tensor_lists.append(tensors)
            specs_lists.append(specs)

        return meta_msgs, tensor_lists, specs_lists

    def _reattach_lifted_tensors(
        self,
        meta_msgs: list[IncomingMessage],
        tensor_lists: list[list[torch.Tensor]],
        specs_lists: list[list[_TensorSpec]],
    ) -> list[IncomingMessage]:
        """Follower path: rebuild messages by stitching specs back into payloads.

        On the entry rank ``_strip_and_lift`` already reattached the lifted
        tensors, so this is a no-op (returns ``meta_msgs`` unchanged).
        Followers, however, see ``meta_msgs`` whose ``data.data`` tree
        still contains placeholder dicts; they need to map ``spec.path``
        back to the freshly received tensor.
        """
        out: list[IncomingMessage] = []
        for msg, tensors, specs in zip(meta_msgs, tensor_lists, specs_lists):
            payload = msg.data
            if payload is None or not hasattr(payload, "data") or not specs:
                out.append(msg)
                continue
            tensor_dict = {spec.path: t for spec, t in zip(specs, tensors)}
            payload_cls = type(payload)
            restored = payload_cls(
                request_id=payload.request_id,
                request=payload.request,
                data=restore_tensors(payload.data, tensor_dict),
            )
            out.append(dataclasses.replace(msg, data=restored))
        return out

    def _allocation_ready_gather(self, *, local_ok: bool) -> list[bool]:
        """Gather per-rank allocation-success flags on the TP CPU group."""
        flags: list[bool] = [False] * self.runner.tp_size
        dist.all_gather_object(
            flags,
            local_ok,
            group=self.runner.tp_group.cpu_group,
        )
        return flags

    def _record_rank_wait_skew(self, timing: _RecvPathTiming) -> None:
        """Optionally synchronize rank recv completion to measure skew.

        This adds a CPU-group gather, so it is gated behind
        ``SGLANG_OMNI_ENCODER_TIMING_DETAIL=1`` and intended for benchmark
        runs, not default serving.
        """
        if not self._sync_recv_timing or self.runner.tp_size <= 1:
            return
        arrived = time.perf_counter()
        arrivals = [0.0] * self.runner.tp_size
        dist.all_gather_object(
            arrivals,
            arrived,
            group=self.runner.tp_group.cpu_group,
        )
        timing.rank_wait_skew_ms = (time.perf_counter() - arrived) * 1000.0
        timing.rank_arrival_skew_ms = (max(arrivals) - min(arrivals)) * 1000.0

    def _recv_messages(
        self,
    ) -> tuple[list[IncomingMessage], BaseException | None]:
        """Drain inbox and broadcast inputs to TP non-entry ranks.

        Never raises — returns ``(messages, error)``. The error is non-None
        if either rank failed during this iteration. Drained messages
        are returned even on entry-rank failure so the scheduler can emit
        request-level errors against them in the unified handshake.
        """
        timing = _RecvPathTiming()
        self._last_recv_timing = timing
        if self.runner.tp_size == 1:
            started = time.perf_counter()
            local, collect_err = self._collect_batch_or_error()
            timing.inbox_admission_ms = (time.perf_counter() - started) * 1000.0
            self._log_memory_mark(
                "after_inbox_admission",
                batch_size=len(local),
            )
            if collect_err is not None or not local:
                return local, collect_err
            try:
                started = time.perf_counter()
                meta_msgs, tensor_lists, specs_lists = self._strip_and_lift(
                    local
                )
                timing.strip_h2d_ms = (time.perf_counter() - started) * 1000.0
                self._log_memory_mark(
                    "after_strip_h2d",
                    batch_size=len(meta_msgs),
                )
            except Exception as exc:  # noqa: BLE001
                return local, exc
            return self._reattach_lifted_tensors(
                meta_msgs,
                tensor_lists,
                specs_lists,
            ), None

        tp = self.runner.tp_group
        src_rank = tp.ranks[0]

        if self.runner.is_entry_rank:
            started = time.perf_counter()
            local, collect_err = self._collect_batch_or_error()
            timing.inbox_admission_ms = (time.perf_counter() - started) * 1000.0
            self._log_memory_mark(
                "after_inbox_admission",
                batch_size=len(local),
            )
            if collect_err is not None:
                started = time.perf_counter()
                broadcast_pyobj(
                    [{"kind": _RECV_ERROR_KIND, "error": repr(collect_err)}],
                    tp.rank, tp.cpu_group, src=src_rank,
                )
                timing.metadata_broadcast_ms = (
                    time.perf_counter() - started
                ) * 1000.0
                self._log_memory_mark(
                    "after_metadata_broadcast",
                    batch_size=len(local),
                )
                return local, collect_err

            try:
                batch_cost = self._batch_cost(local)
                started = time.perf_counter()
                meta_msgs, tensor_lists, specs_lists = self._strip_and_lift(
                    local
                )
                timing.strip_h2d_ms = (time.perf_counter() - started) * 1000.0
                self._log_memory_mark(
                    "after_strip_h2d",
                    batch_size=len(meta_msgs),
                )
            except Exception as exc:  # noqa: BLE001
                started = time.perf_counter()
                broadcast_pyobj(
                    [{"kind": _RECV_ERROR_KIND, "error": repr(exc)}],
                    tp.rank, tp.cpu_group, src=src_rank,
                )
                timing.metadata_broadcast_ms = (
                    time.perf_counter() - started
                ) * 1000.0
                self._log_memory_mark(
                    "after_metadata_broadcast",
                    batch_size=len(local),
                )
                return local, exc

            started = time.perf_counter()
            broadcast_pyobj(
                [meta_msgs, specs_lists, batch_cost],
                tp.rank, tp.cpu_group, src=src_rank,
            )
            timing.metadata_broadcast_ms = (
                time.perf_counter() - started
            ) * 1000.0
            self._log_memory_mark(
                "after_metadata_broadcast",
                batch_size=len(meta_msgs),
            )

            started = time.perf_counter()
            ok_flags = self._allocation_ready_gather(local_ok=True)
            timing.allocation_handshake_ms = (
                time.perf_counter() - started
            ) * 1000.0
            self._log_memory_mark(
                "after_allocation_handshake",
                batch_size=len(meta_msgs),
            )
            if not all(ok_flags):
                return local, RuntimeError(
                    "peer-rank tensor allocation failed"
                )

            started = time.perf_counter()
            for tensor_list in tensor_lists:
                for t in tensor_list:
                    dist.broadcast(t, src=src_rank, group=tp.device_group)
            timing.tensor_broadcast_ms = (
                time.perf_counter() - started
            ) * 1000.0
            self._log_memory_mark(
                "after_tensor_broadcast",
                batch_size=len(meta_msgs),
            )

            self._record_rank_wait_skew(timing)
            return self._reattach_lifted_tensors(
                meta_msgs,
                tensor_lists,
                specs_lists,
            ), None

        # Follower path
        started = time.perf_counter()
        payload = broadcast_pyobj([], tp.rank, tp.cpu_group, src=src_rank)
        timing.metadata_broadcast_ms = (time.perf_counter() - started) * 1000.0
        if (
            payload
            and isinstance(payload[0], dict)
            and payload[0].get("kind") == _RECV_ERROR_KIND
        ):
            return [], RuntimeError(
                f"entry rank failed before metadata broadcast: "
                f"{payload[0]['error']}"
            )
        if len(payload) == 2:
            meta_msgs, specs_lists = payload
            batch_cost = self._batch_cost(meta_msgs)
        else:
            meta_msgs, specs_lists, batch_cost = payload
        self._log_memory_mark(
            "after_metadata_broadcast",
            batch_size=len(meta_msgs),
        )

        placeholders: list[list[torch.Tensor]] = []
        alloc_err: BaseException | None = None
        try:
            started = time.perf_counter()
            decision, reservation = self._reserve_gpu_guard(int(batch_cost))
            if not decision.allowed:
                raise self._format_gpu_guard_error(decision)
            self._active_reservation = reservation
            if reservation is not None:
                logger.info(
                    "encoder_reservation stage=%s decision=reserve "
                    "batch_size=%d batch_cost=%d reserved_bytes=%d "
                    "free_bytes=%s required_bytes=%d",
                    self._stage_name,
                    len(meta_msgs),
                    int(batch_cost),
                    reservation.bytes,
                    decision.free_bytes,
                    decision.required_bytes,
                )
            for specs in specs_lists:
                placeholders.append(
                    [
                        torch.empty(
                            spec.shape,
                            dtype=spec.dtype,
                            device=self.runner.device,
                        )
                        for spec in specs
                    ]
                )
            timing.follower_allocation_ms = (
                time.perf_counter() - started
            ) * 1000.0
            self._log_memory_mark(
                "after_follower_allocation",
                batch_size=len(meta_msgs),
            )
        except Exception as exc:  # noqa: BLE001
            timing.follower_allocation_ms = (
                time.perf_counter() - started
            ) * 1000.0
            self._log_memory_mark(
                "after_follower_allocation",
                batch_size=len(meta_msgs) if "meta_msgs" in locals() else 0,
            )
            alloc_err = exc

        started = time.perf_counter()
        ok_flags = self._allocation_ready_gather(local_ok=alloc_err is None)
        timing.allocation_handshake_ms = (
            time.perf_counter() - started
        ) * 1000.0
        self._log_memory_mark(
            "after_allocation_handshake",
            batch_size=len(meta_msgs),
        )
        if not all(ok_flags):
            return [], (
                alloc_err
                if alloc_err is not None
                else RuntimeError("peer-rank tensor allocation failed")
            )

        started = time.perf_counter()
        for ph_list in placeholders:
            for t in ph_list:
                dist.broadcast(t, src=src_rank, group=tp.device_group)
        timing.tensor_broadcast_ms = (time.perf_counter() - started) * 1000.0
        self._log_memory_mark(
            "after_tensor_broadcast",
            batch_size=len(meta_msgs),
        )

        out = self._reattach_lifted_tensors(
            meta_msgs, placeholders, specs_lists
        )
        self._record_rank_wait_skew(timing)
        return out, None

    # ------------------------------------------------------------------
    # Error emission
    # ------------------------------------------------------------------

    def _emit_error(
        self,
        messages: list[IncomingMessage],
        error: BaseException,
    ) -> None:
        """Emit one ``OutgoingMessage(type="error")`` per drained request.

        SimpleScheduler exposes a single-request ``_emit_error`` helper;
        reusing it here would put a list[IncomingMessage] into
        ``OutgoingMessage.request_id``, which
        ``Stage._drain_outbox_external`` would TypeError on. Iterate
        explicitly so each drained request becomes one HTTP-500.
        """
        if not messages:
            # No request was safely captured — synthesize one anonymous
            # error so the failure does not vanish silently. Stage will
            # discard it because the request_id is empty, but the log
            # captures the cause.
            logger.error(
                "EncoderScheduler iteration failed without drained requests: %s",
                error,
            )
            return
        for msg in messages:
            if msg.request_id in self._aborted_request_ids:
                continue
            self.outbox.put(
                OutgoingMessage(
                    request_id=msg.request_id,
                    type="error",
                    data=error,
                )
            )
