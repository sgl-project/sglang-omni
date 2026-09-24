"""CUDA graphs for fixed-shape Qwen3-TTS incremental Codec decodes."""

from __future__ import annotations

import gc
import logging
import math
import os
import threading
from collections import Counter
from collections.abc import Iterable, Sequence
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
class CapturedIncrementalCodecGraph:
    graph: torch.cuda.CUDAGraph
    static_codes: torch.Tensor
    static_index: torch.Tensor
    waveform: torch.Tensor


@dataclass(slots=True)
class CaptureResourceSet:
    """Strong references retained when capture completion cannot be proven."""

    pool: Any | None
    stream: torch.cuda.Stream | None
    keepalives: list[Any] = field(default_factory=list)


class CaptureFailure(RuntimeError):
    pass


def split_frames_by_width(
    total_frames: int, widths: Iterable[int]
) -> tuple[int, ...] | None:
    """Split total_frames into the widths, largest first; None if it does not divide."""
    remaining = int(total_frames)
    split: list[int] = []
    for width in sorted(widths, reverse=True):
        while remaining >= width:
            split.append(width)
            remaining -= width
    return tuple(split) if remaining == 0 else None


class Qwen3TTSIncrementalCodecCudaGraphRunner:
    """Fixed-shape CUDA Graph runner for incremental Codec decoding.

    One instance is configured for one CUDA device and one lifecycle mode. It
    captures configured ``(fresh_frames, batch_bucket)`` shapes before serving,
    owns their mutable fixed-address code/state buffers, and replays the
    smallest captured batch bucket that fits a compatible cohort.

    COLD and WARM use separate instances because the initial and follow-up
    workers run on different CUDA streams; each mutable buffer set must be
    replayed serially by only one worker. WINDOW is a second instance on the
    initial worker's stream holding the widths a wider decode is split into,
    so a failed capture there leaves the COLD shapes in place.
    """

    WARMUP_ITERATIONS = 3

    def __init__(
        self,
        decoder: Qwen3TTSIncrementalDecoder,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_quantizers: int,
        mode: Literal["cold", "warm", "window"],
        fresh_frames: tuple[int, ...],
        batch_sizes: tuple[int, ...] = (1, 2, 4, 8),
        min_free_gb: float = 3.0,
        enabled: bool = True,
        compile_fresh_frames: Sequence[int] = (),
        arena: Qwen3TTSCodecStateArena,
        stream_priority: int = 0,
    ) -> None:
        self.decoder = decoder
        self.compile_fresh_frames = frozenset((int(f) for f in compile_fresh_frames))
        self.arena = arena
        self.stream_priority = int(stream_priority)
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_quantizers = int(num_quantizers)
        self.mode = str(mode).strip().lower()
        if self.mode not in {"cold", "warm", "window"}:
            raise ValueError(
                "incremental Codec graph mode must be 'cold', 'warm' or 'window'"
            )
        self.fresh_frames = tuple(
            sorted({int(frames) for frames in fresh_frames if int(frames) > 0})
        )
        self.batch_sizes = tuple(
            sorted({int(size) for size in batch_sizes if int(size) > 0})
        )
        if not math.isfinite(float(min_free_gb)) or float(min_free_gb) < 0:
            raise ValueError("incremental Codec graph min_free_gb must be >= 0")
        self.min_free_bytes = int(float(min_free_gb) * 1024**3)
        self.configured = bool(
            enabled
            and self.device.type in {"cuda", "musa"}
            and (self.device.index is not None)
            and (self.num_quantizers > 0)
            and self.fresh_frames
            and self.batch_sizes
        )
        self.enabled = False
        self.disable_reason: str | None = None
        self.owner_pid = os.getpid()
        self.graphs: dict[IncrementalCodecGraphKey, CapturedIncrementalCodecGraph] = {}
        self.capture_complete = False
        self.pool: Any | None = None
        self.capture_stream: torch.cuda.Stream | None = None
        self.memory_stats: dict[str, Any] = {"min_free_bytes": self.min_free_bytes}
        self.retained_capture_resources: list[CaptureResourceSet] = []
        self.replays = 0
        self.replay_failures = 0
        self.misses: Counter[str] = Counter()
        self.graphs_lock = threading.Lock()

    def capture(self) -> None:
        """Capture every configured hot shape before serving readiness."""
        if not self.configured or self.capture_complete:
            return
        self.capture_complete = True
        keys = [
            IncrementalCodecGraphKey(frames, batch_size)
            for frames in self.fresh_frames
            for batch_size in self.batch_sizes
        ]
        temporary: dict[IncrementalCodecGraphKey, CapturedIncrementalCodecGraph] = {}
        pool: Any | None = None
        capture_stream: torch.cuda.Stream | None = None
        try:
            with torch.cuda.device(self.device):
                gc.collect()
                before = self.memory_snapshot()
                self.memory_stats["before"] = before
                self.require_headroom(before["free_bytes"])
                pool = torch.cuda.graph_pool_handle()
                capture_stream = torch.cuda.Stream(
                    device=self.device, priority=self.stream_priority
                )
                for key in sorted(
                    keys,
                    key=lambda item: (item.batch_bucket, item.fresh_frames),
                    reverse=True,
                ):
                    temporary[key] = self.capture_graph(
                        key, pool=pool, capture_stream=capture_stream
                    )
                    torch.cuda.empty_cache()
                    self.require_headroom(
                        torch.cuda.mem_get_info(self.device)[0], key=key
                    )
                after = self.memory_snapshot()
                self.memory_stats["after"] = after
                self.memory_stats["graph_footprint_bytes"] = max(
                    0,
                    after["allocated_bytes"] - before["allocated_bytes"],
                    after["reserved_bytes"] - before["reserved_bytes"],
                )
        except Exception as exc:
            reason = f"capture_failed: {type(exc).__name__}: {exc}"
            self.rollback_capture(
                temporary, pool=pool, capture_stream=capture_stream, reason=reason
            )
            logger.warning(
                "Qwen3-TTS incremental Codec graph capture disabled the %s runner: %s",
                self.mode,
                reason,
                exc_info=True,
            )
            return
        with self.graphs_lock:
            self.graphs = temporary
        self.pool = pool
        self.capture_stream = capture_stream
        self.enabled = bool(self.graphs)
        self.disable_reason = None if self.enabled else "no_graphs_captured"
        logger.info(
            "Qwen3-TTS incremental Codec graphs captured for %s",
            [
                (key.fresh_frames, key.batch_bucket)
                for key in sorted(
                    self.graphs, key=lambda item: (item.fresh_frames, item.batch_bucket)
                )
            ],
        )

    def memory_snapshot(self) -> dict[str, int]:
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }

    def require_headroom(
        self, free_bytes: int, *, key: IncrementalCodecGraphKey | None = None
    ) -> None:
        if int(free_bytes) >= self.min_free_bytes:
            return
        key_text = (
            ""
            if key is None
            else f" after fresh_frames={key.fresh_frames} batch={key.batch_bucket}"
        )
        raise CaptureFailure(
            f"free VRAM {format_bytes_gib(int(free_bytes))} is below {format_bytes_gib(self.min_free_bytes)} headroom{key_text}"
        )

    def capture_graph(
        self,
        key: IncrementalCodecGraphKey,
        *,
        pool: Any,
        capture_stream: torch.cuda.Stream,
    ) -> CapturedIncrementalCodecGraph:
        static_codes = torch.zeros(
            (key.batch_bucket, self.num_quantizers, key.fresh_frames),
            dtype=torch.long,
            device=self.device,
        )
        resources = CaptureResourceSet(
            pool=pool, stream=capture_stream, keepalives=[static_codes]
        )
        graph: torch.cuda.CUDAGraph | None = None
        try:
            self.warmup_capture_shape(key, static_codes, resources)
            current_stream = torch.cuda.current_stream(self.device)
            compiled = key.fresh_frames in self.compile_fresh_frames
            static_index = self.scratch_index(key.batch_bucket)
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
                    state = self.arena.gather_by_index(static_index)
                    waveform = self.decoder.decode(
                        static_codes, state, compiled=compiled
                    )
                    self.arena.scatter_by_index(static_index, state)
            finally:
                torch.cuda.set_stream(current_stream)
            resources.keepalives.append(waveform)
            current_stream.wait_stream(capture_stream)
            capture_stream.synchronize()
            return CapturedIncrementalCodecGraph(
                graph=graph,
                static_codes=static_codes,
                static_index=static_index,
                waveform=waveform,
            )
        except BaseException:
            synchronized = self.retain_capture_resources_if_unsynchronized(resources)
            if synchronized and graph is not None:
                self.reset_graph(graph, context=f"unpublished key {key}")
            raise

    def warmup_capture_shape(
        self,
        key: IncrementalCodecGraphKey,
        static_codes: torch.Tensor,
        resources: CaptureResourceSet,
    ) -> None:
        """Run eager decodes that settle one shape before graph capture."""
        capture_stream = resources.stream
        compiled = key.fresh_frames in self.compile_fresh_frames
        capture_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(capture_stream), torch.inference_mode():
            if compiled:
                trace_state = self.arena.gather_by_index(
                    self.scratch_index(key.batch_bucket)
                )
                resources.keepalives.append(trace_state)
                self.decoder.precompile(static_codes, trace_state)
            for _ in range(self.WARMUP_ITERATIONS):
                warmup_state = self.arena.gather_by_index(
                    self.scratch_index(key.batch_bucket)
                )
                resources.keepalives.append(warmup_state)
                self.decoder.decode(static_codes, warmup_state, compiled=compiled)
        capture_stream.synchronize()
        del resources.keepalives[1:]

    def retain_capture_resources_if_unsynchronized(
        self, resources: CaptureResourceSet
    ) -> bool:
        try:
            resources.stream.synchronize()
        except Exception:
            self.retained_capture_resources.append(resources)
            logger.exception(
                "Qwen3-TTS incremental Codec capture stream could not be synchronized; retaining partial capture resources"
            )
            return False
        return True

    @staticmethod
    def reset_graph(graph: Any, *, context: str) -> None:
        try:
            graph.reset()
        except Exception:
            logger.warning(
                "Failed to reset Qwen3-TTS incremental Codec graph during %s",
                context,
                exc_info=True,
            )

    def rollback_capture(
        self,
        temporary: dict[IncrementalCodecGraphKey, CapturedIncrementalCodecGraph],
        *,
        pool: Any | None,
        capture_stream: torch.cuda.Stream | None,
        reason: str,
    ) -> None:
        with self.graphs_lock:
            self.graphs.clear()
        self.pool = None
        self.capture_stream = None
        self.enabled = False
        self.disable_reason = reason
        if not self.synchronize_device("capture rollback"):
            self.retained_capture_resources.append(
                CaptureResourceSet(
                    pool=pool, stream=capture_stream, keepalives=[temporary]
                )
            )
            return
        self.tear_down_graphs(temporary, context="capture rollback")
        self.retained_capture_resources.clear()

    def synchronize_device(self, context: str) -> bool:
        """Prove every queued replay or capture finished; False keeps their memory alive."""
        try:
            with torch.cuda.device(self.device):
                torch.cuda.synchronize(self.device)
        except RuntimeError as synchronize_exc:
            logger.warning(
                "Qwen3-TTS incremental Codec graph %s synchronize failed; retaining graph buffers for the process lifetime: %s",
                context,
                synchronize_exc,
            )
            return False
        return True

    def tear_down_graphs(
        self,
        graphs: dict[IncrementalCodecGraphKey, CapturedIncrementalCodecGraph],
        *,
        context: str,
    ) -> None:
        for key, captured in graphs.items():
            self.reset_graph(captured.graph, context=f"{context} for {key}")
        graphs.clear()
        gc.collect()
        try:
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        except RuntimeError as cleanup_exc:
            logger.warning(
                "Qwen3-TTS incremental Codec graph %s cleanup failed: %s",
                context,
                cleanup_exc,
            )

    def available_batch_sizes(self, fresh_frames: int) -> tuple[int, ...]:
        """Return published batch buckets for one fresh-frame count."""
        if not self.enabled:
            return ()
        return tuple(
            sorted(
                (
                    key.batch_bucket
                    for key in self.graphs
                    if key.fresh_frames == int(fresh_frames)
                ),
                reverse=True,
            )
        )

    def split_frames(self, total_frames: int) -> tuple[int, ...] | None:
        """Split total_frames into captured widths; None while disabled."""
        if not self.enabled:
            return None
        with self.graphs_lock:
            widths = {key.fresh_frames for key in self.graphs}
        return split_frames_by_width(total_frames, widths)

    def largest_batch_bucket(self) -> int:
        """The widest cohort one replay takes at any captured width; 0 while disabled."""
        if not self.enabled:
            return 0
        with self.graphs_lock:
            return max((key.batch_bucket for key in self.graphs), default=0)

    def scratch_index(self, bucket: int) -> torch.Tensor:
        return torch.full(
            (int(bucket),),
            int(self.arena.scratch_slot),
            dtype=torch.long,
            device=self.device,
        )

    def decode_slots(
        self, codes: torch.Tensor, slots: Sequence[int]
    ) -> torch.Tensor | None:
        """Replay the bucket that fits this cohort directly against the arena.

        Returns the borrowed waveform rows, or None on a graph miss. Rows past
        the cohort read and write the arena's scratch row.
        """
        if os.getpid() != self.owner_pid:
            raise RuntimeError(
                f"Qwen3-TTS incremental Codec graph runner belongs to PID {self.owner_pid}, but was used in PID {os.getpid()}"
            )
        if not self.enabled or not self.graphs:
            with self.graphs_lock:
                self.misses["disabled_or_uncaptured"] += 1
            return None
        self.validate_codes(codes)
        if int(codes.shape[2]) not in self.fresh_frames:
            with self.graphs_lock:
                self.misses["uncaptured_fresh_frames"] += 1
            return None
        batch_size = int(codes.shape[0])
        if batch_size != len(slots):
            raise ValueError("decode_slots needs one slot per code row")
        bucket = next(
            (
                size
                for size in self.batch_sizes
                if size >= batch_size
                and IncrementalCodecGraphKey(int(codes.shape[2]), size) in self.graphs
            ),
            None,
        )
        if bucket is None:
            with self.graphs_lock:
                self.misses["missing_batch_bucket"] += 1
            return None
        entry = self.graphs[IncrementalCodecGraphKey(int(codes.shape[2]), bucket)]
        entry.static_index[:batch_size].copy_(self.arena.stage_index(slots))
        if batch_size < bucket:
            entry.static_index[batch_size:].fill_(int(self.arena.scratch_slot))
            entry.static_codes[batch_size:].zero_()
        entry.static_codes[:batch_size].copy_(codes)
        try:
            entry.graph.replay()
        except Exception as exc:
            self.replay_failures += 1
            reason = f"runtime_replay_failed: {type(exc).__name__}: {exc}"
            self.disable_runtime(reason)
            logger.exception(
                "Qwen3-TTS incremental Codec graph replay disabled the %s runner",
                self.mode,
            )
            raise
        self.replays += 1
        return entry.waveform[:batch_size]

    def validate_codes(self, codes: torch.Tensor) -> None:
        if codes.ndim != 3:
            raise ValueError("incremental Codec graph input must have shape [B, Q, T]")
        if int(codes.shape[0]) < 1:
            raise ValueError("incremental Codec graph input requires at least one row")
        if int(codes.shape[1]) != self.num_quantizers:
            raise ValueError(
                f"incremental Codec graph input must contain {self.num_quantizers} quantizers"
            )
        if codes.dtype != torch.long:
            raise TypeError("incremental Codec graph input must use torch.long")
        if codes.device != self.device:
            raise ValueError(
                f"incremental Codec graph input must be on {self.device}, got {codes.device}"
            )

    def disable_runtime(self, reason: str) -> None:
        self.enabled = False
        self.disable_reason = reason
        if not self.synchronize_device("runtime disable"):
            return
        with self.graphs_lock:
            graphs = dict(self.graphs)
            self.graphs.clear()
        self.pool = None
        self.capture_stream = None
        self.tear_down_graphs(graphs, context="runtime disable")

    def stats(self) -> dict[str, Any]:
        with self.graphs_lock:
            captured_keys = sorted(
                self.graphs, key=lambda key: (key.fresh_frames, key.batch_bucket)
            )
            fallback_counts = dict(sorted(self.misses.items()))
        return {
            "configured": self.configured,
            "enabled": self.enabled,
            "disable_reason": self.disable_reason,
            "binding": {
                "mode": self.mode,
                "device": str(self.device),
                "dtype": str(self.dtype),
                "num_quantizers": self.num_quantizers,
                "owner_pid": self.owner_pid,
            },
            "graph_contract": {
                "fresh_frames": list(self.fresh_frames),
                "batch_sizes": list(self.batch_sizes),
            },
            "build": {
                "capture_complete": self.capture_complete,
                "captured_keys": [
                    {"fresh_frames": key.fresh_frames, "batch_bucket": key.batch_bucket}
                    for key in captured_keys
                ],
            },
            "memory": dict(self.memory_stats),
            "retained_capture_resource_sets": len(self.retained_capture_resources),
            "runtime": {
                "replays": self.replays,
                "replay_failures": self.replay_failures,
                "fallback_counts": fallback_counts,
            },
        }


__all__ = [
    "IncrementalCodecGraphKey",
    "Qwen3TTSIncrementalCodecCudaGraphRunner",
    "split_frames_by_width",
]
