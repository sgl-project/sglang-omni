# SPDX-License-Identifier: Apache-2.0
"""Exact-shape CUDA graphs for the Qwen3-Omni Code2Wav component."""

from __future__ import annotations

import gc
import logging
import math
import os
from collections import Counter
from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GraphKey:
    """One exact Code2Wav input shape, excluding fixed quantizer count."""

    batch_size: int
    frames: int


@dataclass(frozen=True, slots=True)
class Code2WavRunResult:
    """Result metadata for either an exact graph replay or eager fallback.

    A ``cuda_graph`` output is a borrowed static buffer. Before the next replay,
    the caller must either finish every read or enqueue every dependent read and
    copy on the same CUDA stream so replay cannot overtake them. The tensor
    itself must not be retained or consumed concurrently; asynchronous host
    transfer must retain its destination and completion event until
    materialization finishes. This runner deliberately does not clone the
    output.
    """

    output: torch.Tensor
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


class _TorchCudaApi:
    """Small injectable boundary around CUDA-only operations."""

    def device_context(self, device: torch.device) -> AbstractContextManager[Any]:
        return torch.cuda.device(device)

    def memory_stats(self, device: torch.device) -> dict[str, int]:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def new_static_input(
        self, shape: tuple[int, int, int], *, device: torch.device
    ) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.long, device=device)

    def new_stream(self, device: torch.device) -> torch.cuda.Stream:
        return torch.cuda.Stream(device=device)

    def warmup(
        self,
        model: Any,
        static_input: torch.Tensor,
        *,
        iterations: int,
        device: torch.device,
        stream: torch.cuda.Stream,
    ) -> None:
        current_stream = torch.cuda.current_stream(device)
        stream.wait_stream(current_stream)
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(iterations):
                model(static_input)
        current_stream.wait_stream(stream)

    def graph_pool_handle(self) -> Any:
        return torch.cuda.graph_pool_handle()

    def capture(
        self,
        model: Any,
        static_input: torch.Tensor,
        *,
        pool: Any,
        stream: torch.cuda.Stream,
    ) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
        current_stream = torch.cuda.current_stream(static_input.device)
        stream.wait_stream(current_stream)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.inference_mode():
                with torch.cuda.graph(
                    graph,
                    pool=pool,
                    stream=stream,
                    capture_error_mode="thread_local",
                ):
                    static_output = model(static_input)
        finally:
            # torch.cuda.graph.__exit__ calls capture_end before restoring its
            # stream context. If capture_end raises, restore explicitly using
            # the original stream's device-aware identity.
            torch.cuda.set_stream(current_stream)
        current_stream.wait_stream(stream)
        return graph, static_output

    def synchronize(self, device: torch.device) -> None:
        torch.cuda.synchronize(device)

    def is_cuda_tensor(self, tensor: torch.Tensor) -> bool:
        return tensor.is_cuda

    def tensor_device_matches(self, tensor: torch.Tensor, device: torch.device) -> bool:
        return tensor.device == device


class Code2WavCudaGraphRunner:
    """Exact-shape CUDA graph runner for ``[B, Q, T]`` long codes.

    One instance is permanently bound to one model, CUDA device, quantizer
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
        cuda_api: Any,
    ) -> None:
        self._model = model
        self._device = torch.device(device)
        if self._device.type not in ("cuda", "musa") or self._device.index is None:
            raise ValueError("Code2Wav CUDA graphs require a concrete CUDA/MUSA device")
        self._num_quantizers = int(num_quantizers)
        if self._num_quantizers <= 0:
            raise ValueError("Code2Wav CUDA graphs require a positive quantizer count")
        self._graph_keys = graph_keys
        self._tier0_keys = tuple(k for k in graph_keys if k.batch_size == 1)
        self._tier1_keys = tuple(k for k in graph_keys if k.batch_size > 1)
        self._owner_pid = os.getpid()
        self._cuda = cuda_api
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
        cuda_api: Any | None = None,
    ) -> Code2WavCudaGraphRunner:
        """Build the configured serving-reachable serial graphs."""

        runner = cls(
            model,
            device=device,
            num_quantizers=num_quantizers,
            graph_keys=graph_keys,
            cuda_api=_TorchCudaApi() if cuda_api is None else cuda_api,
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
            with self._cuda.device_context(self._device):
                before = self._cuda.memory_stats(self._device)
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
            "Code2Wav CUDA graph runner published %d exact graphs on %s",
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
        try:
            with self._cuda.device_context(self._device):
                pool = self._cuda.graph_pool_handle()
                capture_stream = self._cuda.new_stream(self._device)
                if tier1_keys:
                    previous_footprint = self._footprint_since(before)
                for index, key in enumerate(tier1_keys):
                    self._build_stats["attempted_graph_count"] += 1
                    temporary[key] = self._capture_graph(
                        key,
                        pool=pool,
                        stream=capture_stream,
                    )
                    self._cuda.synchronize(self._device)
                    # Warmup's eager activations linger in the allocator cache
                    # and would count as reserved footprint, dwarfing the pool
                    # itself; release them so the check measures what is kept.
                    self._cuda.empty_cache()
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
                        temporary[key] = self._capture_graph(
                            key,
                            pool=pool,
                            stream=capture_stream,
                        )
                    # Capture, replay and equivalence checks enqueue CUDA
                    # work. Do not make the graph matrix visible until every
                    # key has completed on the bound device.
                    self._cuda.synchronize(self._device)
                    gc.collect()
                    self._cuda.empty_cache()
                    after = self._cuda.memory_stats(self._device)
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
            with self._cuda.device_context(self._device):
                self._cuda.empty_cache()
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
        snapshot = self._cuda.memory_stats(self._device)
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
        static_input = self._cuda.new_static_input(
            (key.batch_size, self._num_quantizers, key.frames),
            device=self._device,
        )
        self._cuda.warmup(
            self._model,
            static_input,
            iterations=self._WARMUP_ITERATIONS,
            device=self._device,
            stream=stream,
        )
        graph, static_output = self._cuda.capture(
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
                self._cuda.synchronize(self._device)
            except Exception as synchronize_exc:
                logger.warning(
                    "Code2Wav CUDA graph rollback synchronize failed: %s",
                    synchronize_exc,
                )
            try:
                self._memory_stats["after"] = self._cuda.memory_stats(self._device)
            except Exception as snapshot_exc:
                logger.warning(
                    "Code2Wav CUDA graph rollback snapshot failed: %s",
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
            with self._cuda.device_context(self._device):
                self._cuda.empty_cache()
                self._memory_stats["after_rollback"] = self._cuda.memory_stats(
                    self._device
                )
        except Exception as cleanup_exc:
            logger.warning(
                "Code2Wav CUDA graph rollback cleanup failed: %s",
                cleanup_exc,
            )
        logger.warning("Code2Wav CUDA graph runner disabled: %s", reason)

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
                "Code2Wav CUDA graph runner/model belongs to PID "
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
        if not self._cuda.is_cuda_tensor(codes):
            raise TypeError("Code2Wav graph input must be a CUDA tensor")
        if codes.dtype != torch.long:
            raise TypeError("Code2Wav graph input must use torch.long")
        if not self._cuda.tensor_device_matches(codes, self._device):
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
            with self._cuda.device_context(self._device):
                self._cuda.empty_cache()
        except Exception as cleanup_exc:
            logger.warning(
                "Code2Wav CUDA graph runtime cleanup failed: %s",
                cleanup_exc,
            )
        logger.exception("Code2Wav CUDA graph replay disabled the runner")

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
