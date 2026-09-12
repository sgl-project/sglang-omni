# SPDX-License-Identifier: Apache-2.0
"""CUDA graphs for fixed-shape Qwen3-TTS incremental Codec decodes."""

from __future__ import annotations

import gc
import logging
import math
import os
import threading
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.utils.gpu_memory import format_bytes_gib

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IncrementalCodecGraphKey:
    """One fixed incremental Codec execution shape."""

    fresh_frames: int
    batch_bucket: int


@dataclass(slots=True)
class _CapturedIncrementalCodecGraph:
    graph: torch.cuda.CUDAGraph
    static_codes: torch.Tensor
    static_index: torch.Tensor
    waveform: torch.Tensor


@dataclass(slots=True)
class _CaptureResourceSet:
    """Strong references retained when capture completion cannot be proven."""

    pool: Any | None
    stream: torch.cuda.Stream | None
    keepalives: list[Any] = field(default_factory=list)


class _CaptureFailure(RuntimeError):
    pass


class Qwen3TTSIncrementalCodecCudaGraphRunner:
    """Fixed-shape CUDA Graph runner for incremental Codec decoding.

    One instance is configured for one CUDA device and one lifecycle mode. It
    captures configured ``(fresh_frames, batch_bucket)`` shapes before serving,
    owns their mutable fixed-address code/state buffers, and replays the
    smallest captured batch bucket that fits a compatible cohort.

    COLD and WARM use separate instances because the initial and follow-up
    workers run on different CUDA streams; each mutable buffer set must be
    replayed serially by only one worker.
    """

    _WARMUP_ITERATIONS = 3

    def __init__(
        self,
        decoder: Qwen3TTSIncrementalDecoder,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_quantizers: int,
        mode: Literal["cold", "warm"],
        fresh_frames: tuple[int, ...],
        batch_sizes: tuple[int, ...] = (1, 2, 4, 8),
        min_free_gb: float = 3.0,
        enabled: bool = True,
        compile_fresh_frames: Sequence[int] = (),
        arena: Qwen3TTSCodecStateArena,
        stream_priority: int = 0,
    ) -> None:
        self._decoder = decoder
        self._compile_fresh_frames = frozenset(int(f) for f in compile_fresh_frames)
        # note (luojiaxuan): bound to an arena, a graph gathers its cohort's
        # rows from the arena, decodes, and scatters the advanced rows back,
        # all inside the replay. The host then only writes slot ids and codes.
        self._arena = arena
        # note (luojiaxuan): CUDA instantiates a captured graph with node
        # priorities taken from the capture stream, not from the stream that
        # replays it, so capture at the priority the decode streams run at.
        self._stream_priority = int(stream_priority)
        self._device = torch.device(device)
        self._dtype = dtype
        self._num_quantizers = int(num_quantizers)
        self._mode = str(mode).strip().lower()
        if self._mode not in {"cold", "warm"}:
            raise ValueError("incremental Codec graph mode must be 'cold' or 'warm'")
        self._fresh_frames = tuple(
            sorted({int(frames) for frames in fresh_frames if int(frames) > 0})
        )
        self._batch_sizes = tuple(
            sorted({int(size) for size in batch_sizes if int(size) > 0})
        )
        if not math.isfinite(float(min_free_gb)) or float(min_free_gb) < 0:
            raise ValueError("incremental Codec graph min_free_gb must be >= 0")
        self._min_free_bytes = int(float(min_free_gb) * (1024**3))
        self._configured = bool(
            enabled
            and self._device.type == "cuda"
            and self._device.index is not None
            and self._num_quantizers > 0
            and self._fresh_frames
            and self._batch_sizes
        )
        self._enabled = False
        self._disable_reason: str | None = None
        self._owner_pid = os.getpid()
        self._graphs: dict[IncrementalCodecGraphKey, _CapturedIncrementalCodecGraph] = (
            {}
        )
        self._capture_complete = False
        self._pool: Any | None = None
        self._capture_stream: torch.cuda.Stream | None = None
        self._memory_stats: dict[str, Any] = {
            "min_free_bytes": self._min_free_bytes,
        }
        self._retained_capture_resources: list[_CaptureResourceSet] = []
        self._replays = 0
        self._replay_failures = 0
        self._misses: Counter[str] = Counter()
        # note (luojiaxuan): stats() runs on whichever worker hits the log
        # interval while another worker may be disabling its runner or
        # recording a first miss; the lock keeps those dict walks consistent.
        self._graphs_lock = threading.Lock()

    def capture(self) -> None:
        """Capture every configured hot shape before serving readiness."""

        if not self._configured or self._capture_complete:
            return
        self._capture_complete = True
        keys = [
            IncrementalCodecGraphKey(frames, batch_size)
            for frames in self._fresh_frames
            for batch_size in self._batch_sizes
        ]
        temporary: dict[IncrementalCodecGraphKey, _CapturedIncrementalCodecGraph] = {}
        pool: Any | None = None
        capture_stream: torch.cuda.Stream | None = None
        try:
            with torch.cuda.device(self._device):
                before = self._memory_snapshot()
                self._memory_stats["before"] = before
                self._require_headroom(before["free_bytes"])
                pool = torch.cuda.graph_pool_handle()
                capture_stream = torch.cuda.Stream(
                    device=self._device, priority=self._stream_priority
                )
                for key in sorted(
                    keys,
                    key=lambda item: (item.batch_bucket, item.fresh_frames),
                    reverse=True,
                ):
                    temporary[key] = self._capture_graph(
                        key,
                        pool=pool,
                        capture_stream=capture_stream,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    self._require_headroom(
                        torch.cuda.mem_get_info(self._device)[0],
                        key=key,
                    )
                after = self._memory_snapshot()
                self._memory_stats["after"] = after
                self._memory_stats["graph_footprint_bytes"] = max(
                    0,
                    after["allocated_bytes"] - before["allocated_bytes"],
                    after["reserved_bytes"] - before["reserved_bytes"],
                )
        except Exception as exc:
            reason = f"capture_failed: {type(exc).__name__}: {exc}"
            self._rollback_capture(
                temporary,
                pool=pool,
                capture_stream=capture_stream,
                reason=reason,
            )
            logger.warning(
                "Qwen3-TTS incremental Codec graph capture disabled the %s runner: %s",
                self._mode,
                reason,
                exc_info=True,
            )
            return

        with self._graphs_lock:
            self._graphs = temporary
        self._pool = pool
        self._capture_stream = capture_stream
        self._enabled = bool(self._graphs)
        self._disable_reason = None if self._enabled else "no_graphs_captured"
        logger.info(
            "Qwen3-TTS incremental Codec graphs captured for %s",
            [
                (key.fresh_frames, key.batch_bucket)
                for key in sorted(
                    self._graphs,
                    key=lambda item: (item.fresh_frames, item.batch_bucket),
                )
            ],
        )

    def _memory_snapshot(self) -> dict[str, int]:
        free_bytes, total_bytes = torch.cuda.mem_get_info(self._device)
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(self._device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(self._device)),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }

    def _require_headroom(
        self,
        free_bytes: int,
        *,
        key: IncrementalCodecGraphKey | None = None,
    ) -> None:
        if int(free_bytes) >= self._min_free_bytes:
            return
        key_text = (
            ""
            if key is None
            else f" after fresh_frames={key.fresh_frames} batch={key.batch_bucket}"
        )
        raise _CaptureFailure(
            "free VRAM "
            f"{format_bytes_gib(int(free_bytes))} is below "
            f"{format_bytes_gib(self._min_free_bytes)} headroom{key_text}"
        )

    def _capture_graph(
        self,
        key: IncrementalCodecGraphKey,
        *,
        pool: Any,
        capture_stream: torch.cuda.Stream,
    ) -> _CapturedIncrementalCodecGraph:
        static_codes = torch.zeros(
            (key.batch_bucket, self._num_quantizers, key.fresh_frames),
            dtype=torch.long,
            device=self._device,
        )
        resources = _CaptureResourceSet(
            pool=pool,
            stream=capture_stream,
            keepalives=[static_codes],
        )
        graph: torch.cuda.CUDAGraph | None = None
        try:
            self._warmup_capture_shape(key, static_codes, resources)
            current_stream = torch.cuda.current_stream(self._device)
            compiled = key.fresh_frames in self._compile_fresh_frames
            static_index = self._scratch_index(key.batch_bucket)
            resources.keepalives.append(static_index)
            graph = torch.cuda.CUDAGraph()
            resources.keepalives.append(graph)
            capture_stream.wait_stream(current_stream)
            try:
                with (
                    torch.inference_mode(),
                    torch.cuda.graph(
                        graph,
                        pool=pool,
                        stream=capture_stream,
                        capture_error_mode="thread_local",
                    ),
                ):
                    state = self._arena.gather_by_index(static_index)
                    waveform = self._decoder.decode(
                        static_codes, state, compiled=compiled
                    )
                    self._arena.scatter_by_index(static_index, state)
            finally:
                torch.cuda.set_stream(current_stream)
            resources.keepalives.append(waveform)
            current_stream.wait_stream(capture_stream)
            capture_stream.synchronize()
            return _CapturedIncrementalCodecGraph(
                graph=graph,
                static_codes=static_codes,
                static_index=static_index,
                waveform=waveform,
            )
        except BaseException:
            synchronized = self._retain_capture_resources_if_unsynchronized(resources)
            if synchronized and graph is not None:
                self._reset_graph(graph, context=f"unpublished key {key}")
            raise

    def _warmup_capture_shape(
        self,
        key: IncrementalCodecGraphKey,
        static_codes: torch.Tensor,
        resources: _CaptureResourceSet,
    ) -> None:
        """Run eager decodes that settle one shape before graph capture."""

        capture_stream = resources.stream
        if key.fresh_frames in self._compile_fresh_frames:
            self._decoder.precompile(
                key.batch_bucket,
                key.fresh_frames,
                num_quantizers=self._num_quantizers,
            )
        capture_stream.wait_stream(torch.cuda.current_stream(self._device))
        with torch.cuda.stream(capture_stream), torch.inference_mode():
            for _ in range(self._WARMUP_ITERATIONS):
                warmup_state = self._arena.gather_by_index(
                    self._scratch_index(key.batch_bucket)
                )
                resources.keepalives.append(warmup_state)
                self._decoder.decode(
                    static_codes,
                    warmup_state,
                    compiled=key.fresh_frames in self._compile_fresh_frames,
                )
        capture_stream.synchronize()
        del resources.keepalives[1:]

    def _retain_capture_resources_if_unsynchronized(
        self,
        resources: _CaptureResourceSet,
    ) -> bool:
        try:
            resources.stream.synchronize()
        except Exception:
            self._retained_capture_resources.append(resources)
            logger.exception(
                "Qwen3-TTS incremental Codec capture stream could not be "
                "synchronized; retaining partial capture resources"
            )
            return False
        return True

    @staticmethod
    def _reset_graph(graph: Any, *, context: str) -> None:
        try:
            graph.reset()
        except Exception:
            logger.warning(
                "Failed to reset Qwen3-TTS incremental Codec graph during %s",
                context,
                exc_info=True,
            )

    def _rollback_capture(
        self,
        temporary: dict[IncrementalCodecGraphKey, _CapturedIncrementalCodecGraph],
        *,
        pool: Any | None,
        capture_stream: torch.cuda.Stream | None,
        reason: str,
    ) -> None:
        with self._graphs_lock:
            self._graphs.clear()
        self._pool = None
        self._capture_stream = None
        self._enabled = False
        self._disable_reason = reason
        if not self._synchronize_device("capture rollback"):
            self._retained_capture_resources.append(
                _CaptureResourceSet(
                    pool=pool,
                    stream=capture_stream,
                    keepalives=[temporary],
                )
            )
            return
        self._tear_down_graphs(temporary, context="capture rollback")
        self._retained_capture_resources.clear()

    def _synchronize_device(self, context: str) -> bool:
        """Prove every queued replay or capture finished; False keeps their memory alive."""
        try:
            with torch.cuda.device(self._device):
                torch.cuda.synchronize(self._device)
        except RuntimeError as synchronize_exc:
            logger.warning(
                "Qwen3-TTS incremental Codec graph %s synchronize failed; "
                "retaining graph buffers for the process lifetime: %s",
                context,
                synchronize_exc,
            )
            return False
        return True

    def _tear_down_graphs(
        self,
        graphs: dict[IncrementalCodecGraphKey, _CapturedIncrementalCodecGraph],
        *,
        context: str,
    ) -> None:
        for key, captured in graphs.items():
            self._reset_graph(captured.graph, context=f"{context} for {key}")
        graphs.clear()
        gc.collect()
        try:
            with torch.cuda.device(self._device):
                torch.cuda.empty_cache()
        except RuntimeError as cleanup_exc:
            logger.warning(
                "Qwen3-TTS incremental Codec graph %s cleanup failed: %s",
                context,
                cleanup_exc,
            )

    def available_batch_sizes(self, fresh_frames: int) -> tuple[int, ...]:
        """Return published batch buckets for one fresh-frame count."""

        if not self._enabled:
            return ()
        return tuple(
            sorted(
                (
                    key.batch_bucket
                    for key in self._graphs
                    if key.fresh_frames == int(fresh_frames)
                ),
                reverse=True,
            )
        )

    def _scratch_index(self, bucket: int) -> torch.Tensor:
        return torch.full(
            (int(bucket),),
            int(self._arena.scratch_slot),
            dtype=torch.long,
            device=self._device,
        )

    def decode_slots(
        self, codes: torch.Tensor, slots: Sequence[int]
    ) -> torch.Tensor | None:
        """Replay the bucket that fits this cohort directly against the arena.

        Returns the borrowed waveform rows, or None on a graph miss. Rows past
        the cohort read and write the arena's scratch row.
        """
        if os.getpid() != self._owner_pid:
            raise RuntimeError(
                "Qwen3-TTS incremental Codec graph runner belongs to PID "
                f"{self._owner_pid}, but was used in PID {os.getpid()}"
            )
        if not self._enabled or not self._graphs:
            with self._graphs_lock:
                self._misses["disabled_or_uncaptured"] += 1
            return None
        self._validate_codes(codes)
        if int(codes.shape[2]) not in self._fresh_frames:
            with self._graphs_lock:
                self._misses["uncaptured_fresh_frames"] += 1
            return None
        batch_size = int(codes.shape[0])
        if batch_size != len(slots):
            raise ValueError("decode_slots needs one slot per code row")
        bucket = next(
            (
                size
                for size in self._batch_sizes
                if size >= batch_size
                and IncrementalCodecGraphKey(int(codes.shape[2]), size) in self._graphs
            ),
            None,
        )
        if bucket is None:
            with self._graphs_lock:
                self._misses["missing_batch_bucket"] += 1
            return None
        entry = self._graphs[IncrementalCodecGraphKey(int(codes.shape[2]), bucket)]
        entry.static_index[:batch_size].copy_(self._arena.stage_index(slots))
        if batch_size < bucket:
            entry.static_index[batch_size:].fill_(int(self._arena.scratch_slot))
            entry.static_codes[batch_size:].zero_()
        entry.static_codes[:batch_size].copy_(codes)
        try:
            entry.graph.replay()
        except Exception as exc:
            self._replay_failures += 1
            reason = f"runtime_replay_failed: {type(exc).__name__}: {exc}"
            self._disable_runtime(reason)
            logger.exception(
                "Qwen3-TTS incremental Codec graph replay disabled the %s runner",
                self._mode,
            )
            raise
        self._replays += 1
        return entry.waveform[:batch_size]

    def _validate_codes(self, codes: torch.Tensor) -> None:
        if codes.ndim != 3:
            raise ValueError("incremental Codec graph input must have shape [B, Q, T]")
        if int(codes.shape[0]) < 1:
            raise ValueError("incremental Codec graph input requires at least one row")
        if int(codes.shape[1]) != self._num_quantizers:
            raise ValueError(
                "incremental Codec graph input must contain "
                f"{self._num_quantizers} quantizers"
            )
        if codes.dtype != torch.long:
            raise TypeError("incremental Codec graph input must use torch.long")
        if codes.device != self._device:
            raise ValueError(
                f"incremental Codec graph input must be on {self._device}, "
                f"got {codes.device}"
            )

    def _disable_runtime(self, reason: str) -> None:
        self._enabled = False
        self._disable_reason = reason
        if not self._synchronize_device("runtime disable"):
            return
        with self._graphs_lock:
            graphs = dict(self._graphs)
            self._graphs.clear()
        self._pool = None
        self._capture_stream = None
        self._tear_down_graphs(graphs, context="runtime disable")

    def stats(self) -> dict[str, Any]:
        with self._graphs_lock:
            captured_keys = sorted(
                self._graphs, key=lambda key: (key.fresh_frames, key.batch_bucket)
            )
            fallback_counts = dict(sorted(self._misses.items()))
        return {
            "configured": self._configured,
            "enabled": self._enabled,
            "disable_reason": self._disable_reason,
            "binding": {
                "mode": self._mode,
                "device": str(self._device),
                "dtype": str(self._dtype),
                "num_quantizers": self._num_quantizers,
                "owner_pid": self._owner_pid,
            },
            "graph_contract": {
                "fresh_frames": list(self._fresh_frames),
                "batch_sizes": list(self._batch_sizes),
            },
            "build": {
                "capture_complete": self._capture_complete,
                "captured_keys": [
                    {
                        "fresh_frames": key.fresh_frames,
                        "batch_bucket": key.batch_bucket,
                    }
                    for key in captured_keys
                ],
            },
            "memory": dict(self._memory_stats),
            "retained_capture_resource_sets": len(self._retained_capture_resources),
            "runtime": {
                "replays": self._replays,
                "replay_failures": self._replay_failures,
                "fallback_counts": fallback_counts,
            },
        }


__all__ = [
    "IncrementalCodecGraphKey",
    "Qwen3TTSIncrementalCodecCudaGraphRunner",
]
