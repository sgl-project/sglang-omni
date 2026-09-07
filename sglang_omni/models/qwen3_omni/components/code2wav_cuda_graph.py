# SPDX-License-Identifier: Apache-2.0
"""Exact-shape device graphs for the Qwen3-Omni Code2Wav component."""

from __future__ import annotations

import contextlib
import gc
import logging
import math
import os
import threading
from collections import Counter
from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

from sglang_omni.platforms import current_platform

logger = logging.getLogger(__name__)

# A non-blocking acquire is both the check and the claim, so two threads cannot
# pass it and race the restore of the transformers mask global.
_MASK_SWAP_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class GraphKey:
    """One exact Code2Wav input shape, excluding fixed quantizer count."""

    batch_size: int
    frames: int


@dataclass(frozen=True, slots=True)
class Code2WavRunResult:
    """Result metadata for either an exact graph replay or eager fallback.

    A replayed output is a borrowed static buffer. Before the next replay, the
    caller must either finish every read or enqueue every dependent read and copy
    on the same device stream so replay cannot overtake them. The tensor itself
    must not be retained or consumed concurrently; asynchronous host transfer must
    retain its destination and completion event until materialization finishes.
    This runner deliberately does not clone the output.
    """

    output: torch.Tensor
    # "cuda_graph" for a replay on any accelerator, "eager" otherwise. The value
    # is spelled for CUDA because it reaches the profiler events, so renaming it
    # would change emitted telemetry on every platform, not just this one.
    execution_mode: str
    key: GraphKey | None
    fallback_reason: str | None


@dataclass(slots=True)
class _CapturedGraph:
    graph: Any
    static_input: torch.Tensor
    static_output: torch.Tensor


class _BuildFailure(RuntimeError):
    pass


@contextlib.contextmanager
def _unpacked_sequence_mask() -> Any:
    """Pin transformers' not-packed mask branch, which it picks by a host read.

    The read aborts a capture, and None is its answer for the single contiguous
    sequence the static input is, so pinning it loses nothing.

    Swapping a module global is only safe while capture stays single-threaded:
    run() replays or falls back to eager and never captures. The lock enforces
    that rather than trusting it.
    """
    from transformers import masking_utils

    if not _MASK_SWAP_LOCK.acquire(blocking=False):
        raise RuntimeError(
            "Code2Wav mask pin is already held; capture must stay single-threaded "
            "because it swaps a transformers global"
        )
    # Reading or replacing the global can raise if transformers renames it, and
    # a leaked lock would refuse every later capture in this process.
    try:
        original = masking_utils.find_packed_sequence_indices
        # Accept any signature: the answer is None regardless, and a mismatch
        # would raise mid-capture and defeat the pin.
        masking_utils.find_packed_sequence_indices = lambda *args, **kwargs: None
        try:
            yield
        finally:
            masking_utils.find_packed_sequence_indices = original
    finally:
        _MASK_SWAP_LOCK.release()


@contextlib.contextmanager
def _xpu_capture_pins() -> Any:
    """Hold the pins XPU's capture needs, and nothing at all elsewhere.

    XPU ends a capture on two host reads: SDPA's backend choice and transformers'
    packed-sequence probe. Both pins are XPU-only on purpose -- CUDA, ROCm, MUSA and
    NPU captured fine without them, so they keep the interpreter untouched and the
    numerics they already had. Guarding here rather than relying on the pins being
    no-ops off XPU keeps that a property of the code, not of two hooks agreeing.
    """
    if not current_platform.is_xpu():
        yield
        return
    with current_platform.graph_capture_attention(), _unpacked_sequence_mask():
        yield


class _TorchDeviceApi:
    """Injectable boundary that resolves each call's device module from its
    device, so one boundary serves CUDA, MUSA and XPU."""

    @staticmethod
    def _module(device: torch.device) -> Any:
        return torch.get_device_module(device)

    def graph_backend(self, device: torch.device) -> Any | None:
        """The platform's recorder for this device, or None if it names none."""
        return current_platform.get_device_graph_backend(device)

    def device_context(self, device: torch.device) -> AbstractContextManager[Any]:
        return self._module(device).device(device)

    def memory_stats(self, device: torch.device) -> dict[str, int]:
        module = self._module(device)
        free_bytes, total_bytes = module.mem_get_info(device)
        return {
            "allocated_bytes": int(module.memory_allocated(device)),
            "reserved_bytes": int(module.memory_reserved(device)),
            "max_reserved_bytes": int(module.max_memory_reserved(device)),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }

    def empty_cache(self, device: torch.device) -> None:
        self._module(device).empty_cache()

    def new_static_input(
        self, shape: tuple[int, int, int], *, device: torch.device
    ) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.long, device=device)

    def new_stream(self, device: torch.device) -> Any:
        return self._module(device).Stream(device=device)

    def warmup(
        self,
        model: Any,
        static_input: torch.Tensor,
        *,
        iterations: int,
        device: torch.device,
        stream: Any,
    ) -> None:
        module = self._module(device)
        current_stream = module.current_stream(device)
        stream.wait_stream(current_stream)
        # Warmup shares the pins, or it warms a kernel the capture will not use.
        with (
            module.stream(stream),
            torch.inference_mode(),
            _xpu_capture_pins(),
        ):
            for _ in range(iterations):
                model(static_input)
        current_stream.wait_stream(stream)

    def graph_pool_handle(self, device: torch.device) -> Any:
        return self._module(device).graph_pool_handle()

    def capture(
        self,
        model: Any,
        static_input: torch.Tensor,
        *,
        pool: Any,
        stream: Any,
    ) -> tuple[Any, torch.Tensor]:
        device = static_input.device
        module = self._module(device)
        current_stream = module.current_stream(device)
        stream.wait_stream(current_stream)
        # Non-None because the runner refused construction otherwise.
        backend = self.graph_backend(device)
        try:
            with torch.inference_mode(), _xpu_capture_pins():
                with backend.capture(
                    pool=pool, stream=stream, thread_local_errors=True
                ) as graph:
                    static_output = model(static_input)
        finally:
            # graph.__exit__ calls capture_end before restoring its stream
            # context. If capture_end raises, restore explicitly using the
            # original stream's device-aware identity.
            module.set_stream(current_stream)
        current_stream.wait_stream(stream)
        return graph, static_output

    def synchronize(self, device: torch.device) -> None:
        self._module(device).synchronize(device)

    def is_accelerator_tensor(self, tensor: torch.Tensor, device: torch.device) -> bool:
        return tensor.device.type == device.type

    def tensor_device_matches(self, tensor: torch.Tensor, device: torch.device) -> bool:
        return tensor.device == device


class Code2WavCudaGraphRunner:
    """Exact-shape device graph runner for ``[B, Q, T]`` long codes.

    One instance is permanently bound to one model, device, quantizer
    count, ``torch.long`` input dtype, and owner process. ``batch_size == 1``
    keys form an atomic tier with the original semantics: any failure there
    disables the complete runner and leaves no partial matrix published.
    ``batch_size > 1`` keys are best-effort. All keys share one mempool, whose
    total stays near the largest member's peak instead of paying that peak
    once per pool; because pool memory is only reclaimable as a whole, the
    retry unit is a whole capture attempt. Each attempt captures the batched
    keys first — largest-first, each followed by a budget check while the pool
    holds nothing serving depends on — then closes with the atomic tier, so an
    oversized batched graph can never take down the single-request tier that
    serving already relies on.
    """

    _WARMUP_ITERATIONS = 3

    def __init__(
        self,
        model: Any,
        *,
        device: str | torch.device,
        num_quantizers: int,
        graph_keys: tuple[GraphKey, ...],
        device_api: Any,
    ) -> None:
        self._model = model
        self._device = torch.device(device)
        self._device_api = device_api
        if self._device.index is None:
            raise ValueError(
                f"Code2Wav graphs require a concrete device, got {self._device}"
            )
        # The platform owns which devices can record a graph, so ask it rather
        # than keeping a second list here that has to agree. Failing at
        # construction also keeps an unusable device out of the capture attempt,
        # where the exception would be absorbed as one more eager fallback.
        if self._device_api.graph_backend(self._device) is None:
            raise ValueError(
                f"{current_platform.device_type} names no device graph backend, "
                f"so Code2Wav cannot capture on {self._device}"
            )
        self._num_quantizers = int(num_quantizers)
        if self._num_quantizers <= 0:
            raise ValueError("Code2Wav graphs require a positive quantizer count")
        self._graph_keys = graph_keys
        self._tier0_keys = tuple(k for k in graph_keys if k.batch_size == 1)
        self._tier1_keys = tuple(k for k in graph_keys if k.batch_size > 1)
        self._owner_pid = os.getpid()
        self._graphs: dict[GraphKey, _CapturedGraph] = {}
        # Note (ruoyu): the scheduler reads the published sizes several times
        # per step, so they are cached and refreshed where the key set changes
        # (publish, rollback, runtime disable) instead of rescanned per call.
        self._sizes_by_frames: dict[int, tuple[int, ...]] = {}
        self._pool: Any | None = None
        self._capture_stream: Any | None = None
        self._enabled = False
        self._disable_reason: str | None = None
        self._build_stats: dict[str, Any] = {
            "attempted_graph_count": 0,
            "published_graph_count": 0,
        }
        self._memory_stats: dict[str, Any] = {"total_gpu_memory_fraction": None}
        self._fallback_counts: Counter[str] = Counter()
        self._graph_replays = 0
        self._replay_failures = 0

    @classmethod
    def build(
        cls,
        model: Any,
        *,
        device: str | torch.device,
        num_quantizers: int,
        total_gpu_memory_fraction: float | None,
        graph_keys: tuple[GraphKey, ...],
        device_api: Any | None = None,
    ) -> Code2WavCudaGraphRunner:
        """Build the configured serving-reachable serial graphs."""

        runner = cls(
            model,
            device=device,
            num_quantizers=num_quantizers,
            graph_keys=graph_keys,
            device_api=_TorchDeviceApi() if device_api is None else device_api,
        )
        runner._build(total_gpu_memory_fraction)
        return runner

    def _build(self, total_gpu_memory_fraction: float | None) -> None:
        fraction = self._valid_fraction(total_gpu_memory_fraction)
        if fraction is None:
            self._disable_reason = "invalid_total_gpu_memory_fraction"
            return
        self._memory_stats["total_gpu_memory_fraction"] = fraction

        tier1_info: dict[str, Any] = {
            "attempted_key_count": len(self._tier1_keys),
            "published_key_count": 0,
            "attempts": 0,
            "skipped_keys": [],
            "disable_reason": None,
            "per_key_footprint_bytes": {},
        }
        if self._tier1_keys:
            self._memory_stats["tier1"] = tier1_info

        try:
            with self._device_api.device_context(self._device):
                before = self._device_api.memory_stats(self._device)
        except Exception as exc:
            self._rollback_build(
                temporary={},
                reason=f"capture_failed: {type(exc).__name__}: {exc}",
            )
            return
        self._memory_stats["before"] = before
        stage_budget = int(before["total_bytes"] * fraction)
        loaded_model_footprint = before["allocated_bytes"]
        graph_budget = max(0, stage_budget - loaded_model_footprint)
        self._memory_stats.update(
            {
                "stage_budget_bytes": stage_budget,
                "loaded_model_footprint_bytes": loaded_model_footprint,
                "graph_budget_bytes": graph_budget,
            }
        )

        remaining = list(self._priority_order(self._tier1_keys))
        while True:
            if remaining:
                if tier1_info["attempts"] >= self._TIER1_MAX_ATTEMPTS:
                    remaining = []
                else:
                    tier1_info["attempts"] += 1
            outcome, payload = self._capture_attempt(
                before=before,
                graph_budget=graph_budget,
                tier1_keys=tuple(remaining),
                tier1_info=tier1_info,
            )
            if outcome == "shrink":
                remaining = payload
                continue
            if outcome == "disable":
                temporary, reason = payload
                self._rollback_build(temporary=temporary, reason=reason)
                return
            temporary, pool, capture_stream = payload
            break

        self._pool = pool
        self._capture_stream = capture_stream
        self._graphs = {
            key: temporary[key] for key in self._graph_keys if key in temporary
        }
        sizes_by_frames: dict[int, set[int]] = {}
        for key in self._graphs:
            sizes_by_frames.setdefault(key.frames, set()).add(key.batch_size)
        self._sizes_by_frames = {
            frames: tuple(sorted(sizes, reverse=True))
            for frames, sizes in sizes_by_frames.items()
        }
        self._build_stats["published_graph_count"] = len(self._graphs)
        if self._tier1_keys:
            tier1_info["published_key_count"] = sum(
                1 for key in self._graphs if key.batch_size > 1
            )
            tier1_info["skipped_keys"] = [
                {"batch_size": key.batch_size, "frames": key.frames}
                for key in self._tier1_keys
                if key not in self._graphs
            ]
            if tier1_info["skipped_keys"]:
                logger.warning(
                    "Code2Wav tier-1 graphs published %d/%d keys; skipped: %s",
                    tier1_info["published_key_count"],
                    len(self._tier1_keys),
                    tier1_info["skipped_keys"],
                )
        self._enabled = True
        logger.info(
            "Code2Wav device graph runner published %d exact graphs on %s",
            len(self._graphs),
            self._device,
        )

    # Retries re-capture a strictly smaller key set, so this bound is only a
    # backstop against footprint measurements that never stabilize.
    _TIER1_MAX_ATTEMPTS = 6

    def _capture_attempt(
        self,
        *,
        before: dict[str, int],
        graph_budget: int,
        tier1_keys: tuple[GraphKey, ...],
        tier1_info: dict[str, Any],
    ) -> tuple[str, Any]:
        """Capture every requested key into one fresh shared pool.

        Tier-1 keys go first, largest-first so the pool's peak blocks are laid
        down once, each followed by a budget check while the pool holds
        nothing serving depends on. A violation by the very first key (or by
        the combined footprint after tier 0, whose members are too small to be
        worth shrinking individually) excludes the largest remaining
        batch-size class — small keys share the peak blocks, so only dropping
        a class meaningfully shrinks the pool. Non-capacity failures on a
        tier-1 key abandon the tier: shrinking cannot fix a correctness
        problem, and the atomic tier stays published either way.
        """
        temporary: dict[GraphKey, _CapturedGraph] = {}
        pool: Any | None = None
        capture_stream: Any | None = None
        violation_index: int | None = None
        combined_violation = False
        tier1_abandoned = False
        error_reason: str | None = None
        tier0_started = False
        # (key, phase) together: two variables drifted apart once already, with a
        # tier reporting "capturing" for a failure in the measurement after it.
        capturing: tuple[GraphKey, str] | None = None
        try:
            with self._device_api.device_context(self._device):
                pool = self._device_api.graph_pool_handle(self._device)
                capture_stream = self._device_api.new_stream(self._device)
                if tier1_keys:
                    previous_footprint = self._footprint_since(before)
                for index, key in enumerate(tier1_keys):
                    self._build_stats["attempted_graph_count"] += 1
                    capturing = (key, "capturing")
                    temporary[key] = self._capture_graph(
                        key,
                        pool=pool,
                        stream=capture_stream,
                    )
                    capturing = (key, "measuring after")
                    self._device_api.synchronize(self._device)
                    # Warmup's eager activations linger in the allocator cache
                    # and would count as reserved footprint, dwarfing the pool
                    # itself; release them so the check measures what is kept.
                    self._device_api.empty_cache(self._device)
                    footprint = self._footprint_since(before)
                    tier1_info["per_key_footprint_bytes"][self._key_name(key)] = (
                        footprint - previous_footprint
                    )
                    if footprint > graph_budget:
                        violation_index = index
                        break
                    previous_footprint = footprint
                if violation_index is None:
                    tier0_started = True
                    for key in self._priority_order(self._tier0_keys):
                        self._build_stats["attempted_graph_count"] += 1
                        capturing = (key, "capturing")
                        temporary[key] = self._capture_graph(
                            key,
                            pool=pool,
                            stream=capture_stream,
                        )
                        capturing = (key, "measuring after")
                    # Capture, replay and equivalence checks enqueue device
                    # work. Do not make the graph matrix visible until every
                    # key has completed on the bound device.
                    self._device_api.synchronize(self._device)
                    gc.collect()
                    self._device_api.empty_cache(self._device)
                    after = self._device_api.memory_stats(self._device)
                    self._memory_stats["after"] = after
                    graph_footprint = max(
                        0,
                        after["allocated_bytes"] - before["allocated_bytes"],
                        after["reserved_bytes"] - before["reserved_bytes"],
                    )
                    self._memory_stats["graph_footprint_bytes"] = graph_footprint
                    if graph_footprint > graph_budget:
                        if tier1_keys:
                            combined_violation = True
                        else:
                            raise _BuildFailure(
                                f"memory_budget_exceeded: graph footprint "
                                f"{graph_footprint} exceeds budget "
                                f"{graph_budget}",
                            )
        except torch.OutOfMemoryError as exc:
            if not tier1_keys:
                error_reason = f"capture_failed: {type(exc).__name__}: {exc}"
            elif tier0_started:
                combined_violation = True
            else:
                violation_index = len(temporary)
        except Exception as exc:
            reason = (
                str(exc)
                if isinstance(exc, _BuildFailure)
                else f"capture_failed: {type(exc).__name__}: {exc}"
            )
            if not isinstance(exc, _BuildFailure):
                # A key owns its capture and the measurement after it, so name
                # the key either way and let the phase say which half failed.
                logger.warning(
                    "Code2Wav graph build failed on %s while %s",
                    self._device,
                    (
                        f"{capturing[1]} key={self._key_name(capturing[0])}"
                        if capturing
                        else "setting up the capture pool"
                    ),
                    exc_info=True,
                )
            if tier1_keys and not tier0_started:
                tier1_info["disable_reason"] = reason
                tier1_abandoned = True
            else:
                error_reason = reason

        if error_reason is not None:
            return "disable", (temporary, error_reason)
        if violation_index is None and not combined_violation and not tier1_abandoned:
            return "published", (temporary, pool, capture_stream)

        # Tear the whole attempt down: pool memory frees only once every
        # graph captured into it is gone.
        temporary.clear()
        pool = None
        capture_stream = None
        gc.collect()
        try:
            with self._device_api.device_context(self._device):
                self._device_api.empty_cache(self._device)
        except Exception as cleanup_exc:
            logger.warning(
                "Code2Wav graph attempt rollback cleanup failed: %s",
                cleanup_exc,
            )
        if tier1_abandoned:
            return "shrink", []
        remaining = list(tier1_keys)
        if combined_violation or violation_index == 0:
            oversized_batch = remaining[0].batch_size
            remaining = [key for key in remaining if key.batch_size < oversized_batch]
        else:
            remaining = remaining[:violation_index]
        return "shrink", remaining

    def _footprint_since(self, before: dict[str, int]) -> int:
        snapshot = self._device_api.memory_stats(self._device)
        return max(
            0,
            snapshot["allocated_bytes"] - before["allocated_bytes"],
            snapshot["reserved_bytes"] - before["reserved_bytes"],
        )

    @staticmethod
    def _priority_order(keys: tuple[GraphKey, ...]) -> tuple[GraphKey, ...]:
        # Largest first: the biggest graph lays down the pool's peak blocks so
        # later captures reuse them instead of growing the pool.
        return tuple(sorted(keys, key=lambda k: (k.batch_size, k.frames), reverse=True))

    @staticmethod
    def _key_name(key: GraphKey) -> str:
        return f"b{key.batch_size}t{key.frames}"

    def available_batch_sizes(self, frames: int) -> tuple[int, ...]:
        """Batch sizes with a published graph for this window length, largest
        first; the scheduler decomposes coalesced batches against this."""
        return self._sizes_by_frames.get(int(frames), ())

    def _capture_graph(
        self,
        key: GraphKey,
        *,
        pool: Any,
        stream: Any,
    ) -> _CapturedGraph:
        static_input = self._device_api.new_static_input(
            (key.batch_size, self._num_quantizers, key.frames),
            device=self._device,
        )
        self._device_api.warmup(
            self._model,
            static_input,
            iterations=self._WARMUP_ITERATIONS,
            device=self._device,
            stream=stream,
        )
        graph, static_output = self._device_api.capture(
            self._model,
            static_input,
            pool=pool,
            stream=stream,
        )
        with torch.inference_mode():
            eager_output = self._model(static_input).detach().clone()
            graph.replay()
        self._verify_equivalence(
            key=key,
            eager_output=eager_output,
            graph_output=static_output,
        )
        return _CapturedGraph(graph, static_input, static_output)

    @staticmethod
    def _valid_fraction(value: float | None) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            fraction = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            return None
        return fraction

    @staticmethod
    def _verify_equivalence(
        *,
        key: GraphKey,
        eager_output: torch.Tensor,
        graph_output: torch.Tensor,
    ) -> None:
        if not (
            eager_output.shape == graph_output.shape
            and bool(torch.isfinite(eager_output).all().item())
            and bool(torch.isfinite(graph_output).all().item())
            and torch.equal(eager_output, graph_output)
        ):
            raise _BuildFailure(
                f"equivalence_failed: {key}: eager and graph outputs differ"
            )

    def _rollback_build(
        self,
        *,
        temporary: dict[GraphKey, _CapturedGraph],
        reason: str,
    ) -> None:
        if "after" not in self._memory_stats:
            try:
                self._device_api.synchronize(self._device)
            except Exception as synchronize_exc:
                logger.warning(
                    "Code2Wav device graph rollback synchronize failed: %s",
                    synchronize_exc,
                )
            try:
                self._memory_stats["after"] = self._device_api.memory_stats(
                    self._device
                )
            except Exception as snapshot_exc:
                logger.warning(
                    "Code2Wav device graph rollback snapshot failed: %s",
                    snapshot_exc,
                )
        self._graphs.clear()
        self._sizes_by_frames = {}
        temporary.clear()
        self._pool = None
        self._capture_stream = None
        self._enabled = False
        self._disable_reason = reason
        gc.collect()
        try:
            with self._device_api.device_context(self._device):
                self._device_api.empty_cache(self._device)
                self._memory_stats["after_rollback"] = self._device_api.memory_stats(
                    self._device
                )
        except Exception as cleanup_exc:
            logger.warning(
                "Code2Wav device graph rollback cleanup failed: %s",
                cleanup_exc,
            )
        logger.warning("Code2Wav device graph runner disabled: %s", reason)

    def run(
        self,
        codes: torch.Tensor,
        *,
        eligible: bool = True,
    ) -> Code2WavRunResult:
        """Replay an exact graph or eagerly execute with a stable reason.

        Graph outputs are borrowed and valid only until the next graph replay;
        callers must serialize replay through trim and D2H consumption.
        """

        current_pid = os.getpid()
        if current_pid != self._owner_pid:
            raise RuntimeError(
                "Code2Wav device graph runner/model belongs to PID "
                f"{self._owner_pid}, but was used in PID {current_pid}; it must "
                "be rebuilt in a spawned process before inference"
            )
        if not self._enabled:
            return self._eager(codes, key=None, reason="disabled")
        if not eligible:
            return self._eager(codes, key=None, reason="ineligible")
        self._validate_codes(codes)

        key = GraphKey(
            batch_size=int(codes.shape[0]),
            frames=int(codes.shape[2]),
        )
        captured = self._graphs.get(key)
        if captured is None:
            return self._eager(codes, key=key, reason="key_miss")

        try:
            captured.static_input.copy_(codes)
            captured.graph.replay()
        except Exception as exc:
            self._replay_failures += 1
            reason = f"runtime_replay_failed: {type(exc).__name__}: {exc}"
            # Drop the last local graph reference before cleanup releases its pool.
            captured = None
            self._disable_runtime(reason)
            raise
        self._graph_replays += 1
        return Code2WavRunResult(
            output=captured.static_output,
            execution_mode="cuda_graph",
            key=key,
            fallback_reason=None,
        )

    def _validate_codes(self, codes: torch.Tensor) -> None:
        if not self._device_api.is_accelerator_tensor(codes, self._device):
            raise TypeError(
                f"Code2Wav graph input must be on device type "
                f"{self._device.type!r}, got {codes.device.type!r}"
            )
        if codes.dtype != torch.long:
            raise TypeError("Code2Wav graph input must use torch.long")
        if not self._device_api.tensor_device_matches(codes, self._device):
            raise ValueError(f"Code2Wav graph input must be on {self._device}")
        if codes.ndim != 3:
            raise ValueError("Code2Wav graph input must have shape [B, Q, T]")
        if int(codes.shape[1]) != self._num_quantizers:
            raise ValueError(
                f"Code2Wav graph input must contain {self._num_quantizers} quantizers"
            )

    def _eager(
        self,
        codes: torch.Tensor,
        *,
        key: GraphKey | None,
        reason: str,
    ) -> Code2WavRunResult:
        self._fallback_counts[reason] += 1
        with torch.inference_mode():
            output = self._model(codes)
        return Code2WavRunResult(
            output=output,
            execution_mode="eager",
            key=key,
            fallback_reason=reason,
        )

    def _disable_runtime(self, reason: str) -> None:
        self._graphs.clear()
        self._sizes_by_frames = {}
        self._pool = None
        self._capture_stream = None
        self._enabled = False
        self._disable_reason = reason
        gc.collect()
        try:
            with self._device_api.device_context(self._device):
                self._device_api.empty_cache(self._device)
        except Exception as cleanup_exc:
            logger.warning(
                "Code2Wav device graph runtime cleanup failed: %s",
                cleanup_exc,
            )
        logger.exception("Code2Wav device graph replay disabled the runner")

    def stats(self) -> dict[str, Any]:
        """Return a strict JSON-safe snapshot of build and runtime state."""

        return {
            "enabled": self._enabled,
            "disable_reason": self._disable_reason,
            "binding": {
                "device": str(self._device),
                "num_quantizers": self._num_quantizers,
                "input_dtype": "torch.long",
                "owner_pid": self._owner_pid,
            },
            "graph_contract": {
                "keys": [
                    {
                        "batch_size": key.batch_size,
                        "frames": key.frames,
                    }
                    for key in self._graph_keys
                ],
            },
            "build": deepcopy(self._build_stats),
            "memory": deepcopy(self._memory_stats),
            "runtime": {
                "graph_replays": self._graph_replays,
                "replay_failures": self._replay_failures,
                "fallback_counts": dict(sorted(self._fallback_counts.items())),
            },
        }


__all__ = [
    "Code2WavCudaGraphRunner",
    "Code2WavRunResult",
    "GraphKey",
]
