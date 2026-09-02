# SPDX-License-Identifier: Apache-2.0
"""CUDA graphs for fixed-shape Qwen3-TTS incremental Codec decodes."""

from __future__ import annotations

import gc
import logging
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from sglang_omni.models.qwen3_tts.incremental_codec import (
    Qwen3TTSIncrementalCodecState,
    Qwen3TTSIncrementalDecoder,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IncrementalCodecGraphKey:
    """One fixed incremental Codec execution shape."""

    fresh_frames: int
    batch_bucket: int


@dataclass(frozen=True, slots=True)
class IncrementalCodecGraphResult:
    """Borrowed outputs from one graph replay.

    The caller must enqueue every state scatter and waveform read before the
    next replay of the same runner. Only the first ``batch_size`` rows are real;
    padded rows are never returned.
    """

    waveform: torch.Tensor
    state: Qwen3TTSIncrementalCodecState


@dataclass(slots=True)
class _CapturedIncrementalCodecGraph:
    graph: torch.cuda.CUDAGraph
    static_codes: torch.Tensor
    input_state: Qwen3TTSIncrementalCodecState
    output_state: Qwen3TTSIncrementalCodecState
    waveform: torch.Tensor


@dataclass(slots=True)
class _CaptureResourceSet:
    """Strong references retained when capture completion cannot be proven."""

    pool: Any | None
    stream: torch.cuda.Stream | None
    keepalives: list[Any] = field(default_factory=list)


class _CaptureFailure(RuntimeError):
    pass


def _make_output_state_shell(
    state: Qwen3TTSIncrementalCodecState,
) -> Qwen3TTSIncrementalCodecState:
    """Create an output shell that initially references the input tensors."""

    return Qwen3TTSIncrementalCodecState(
        frame_position=state.frame_position,
        transformer_context_length=state.transformer_context_length,
        frame_positions=state.frame_positions,
        transformer_keys=dict(state.transformer_keys),
        transformer_values=dict(state.transformer_values),
        conv_histories=dict(state.conv_histories),
        transconv_overlaps=dict(state.transconv_overlaps),
    )


def _slice_state_rows(
    state: Qwen3TTSIncrementalCodecState,
    rows: int,
) -> Qwen3TTSIncrementalCodecState:
    if state.frame_positions is None:
        raise RuntimeError("incremental Codec graph output is missing frame positions")
    return Qwen3TTSIncrementalCodecState(
        frame_position=0,
        transformer_context_length=state.transformer_context_length,
        frame_positions=state.frame_positions[:rows],
        transformer_keys={
            key: value[:rows] for key, value in state.transformer_keys.items()
        },
        transformer_values={
            key: value[:rows] for key, value in state.transformer_values.items()
        },
        conv_histories={
            key: value[:rows] for key, value in state.conv_histories.items()
        },
        transconv_overlaps={
            key: value[:rows] for key, value in state.transconv_overlaps.items()
        },
    )


def _copy_tensor_rows(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    rows: int,
    name: str,
) -> None:
    expected = (rows, *destination.shape[1:])
    if tuple(source.shape) != expected:
        raise RuntimeError(
            f"incremental Codec graph expected {expected} for {name}, "
            f"got {tuple(source.shape)}"
        )
    if source.dtype != destination.dtype or source.device != destination.device:
        raise RuntimeError(
            f"incremental Codec graph expected {destination.device}/{destination.dtype} "
            f"for {name}, got {source.device}/{source.dtype}"
        )
    destination[:rows].copy_(source)
    if rows < int(destination.shape[0]):
        destination[rows:].zero_()


def _copy_state_rows(
    destination: Qwen3TTSIncrementalCodecState,
    source: Qwen3TTSIncrementalCodecState,
    *,
    rows: int,
) -> None:
    if destination.frame_positions is None or source.frame_positions is None:
        raise RuntimeError("incremental Codec graph requires per-row frame positions")
    _copy_tensor_rows(
        destination.frame_positions,
        source.frame_positions,
        rows=rows,
        name="frame_positions",
    )

    mappings = (
        ("transformer_keys", destination.transformer_keys, source.transformer_keys),
        (
            "transformer_values",
            destination.transformer_values,
            source.transformer_values,
        ),
        ("conv_histories", destination.conv_histories, source.conv_histories),
        (
            "transconv_overlaps",
            destination.transconv_overlaps,
            source.transconv_overlaps,
        ),
    )
    for label, destination_mapping, source_mapping in mappings:
        if destination_mapping.keys() != source_mapping.keys():
            raise RuntimeError(
                f"incremental Codec graph {label} keys do not match the state spec"
            )
        for key, destination_tensor in destination_mapping.items():
            _copy_tensor_rows(
                destination_tensor,
                source_mapping[key],
                rows=rows,
                name=f"{label}.{key}",
            )


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
    ) -> None:
        self._decoder = decoder
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
                capture_stream = torch.cuda.Stream(device=self._device)
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
            f"{int(free_bytes) / 1024**3:.2f} GiB is below "
            f"{self._min_free_bytes / 1024**3:.2f} GiB headroom{key_text}"
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
            input_state = self._decoder.init_state(
                key.batch_bucket,
                device=self._device,
                dtype=self._dtype,
            )
            output_state = _make_output_state_shell(input_state)
            graph = torch.cuda.CUDAGraph()
            resources.keepalives.extend((input_state, output_state, graph))
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
                    waveform = self._decoder.decode(static_codes, output_state)
            finally:
                torch.cuda.set_stream(current_stream)
            resources.keepalives.append(waveform)
            current_stream.wait_stream(capture_stream)
            capture_stream.synchronize()
            return _CapturedIncrementalCodecGraph(
                graph=graph,
                static_codes=static_codes,
                input_state=input_state,
                output_state=output_state,
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
        if capture_stream is None:
            raise RuntimeError("incremental Codec graph warmup requires a CUDA stream")
        capture_stream.wait_stream(torch.cuda.current_stream(self._device))
        with torch.cuda.stream(capture_stream), torch.inference_mode():
            for _ in range(self._WARMUP_ITERATIONS):
                warmup_state = self._decoder.init_state(
                    key.batch_bucket,
                    device=self._device,
                    dtype=self._dtype,
                )
                resources.keepalives.append(warmup_state)
                self._decoder.decode(static_codes, warmup_state)
        capture_stream.synchronize()
        del resources.keepalives[1:]

    def _retain_capture_resources_if_unsynchronized(
        self,
        resources: _CaptureResourceSet,
    ) -> bool:
        capture_stream = resources.stream
        if capture_stream is None:
            self._retained_capture_resources.append(resources)
            return False
        try:
            capture_stream.synchronize()
        except BaseException:
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
        self._graphs.clear()
        self._pool = None
        self._capture_stream = None
        self._enabled = False
        self._disable_reason = reason
        synchronized = False
        try:
            with torch.cuda.device(self._device):
                torch.cuda.synchronize(self._device)
            synchronized = True
        except RuntimeError as synchronize_exc:
            logger.warning(
                "Qwen3-TTS incremental Codec graph rollback synchronize failed; "
                "retaining capture resources for the process lifetime: %s",
                synchronize_exc,
            )
        if not synchronized:
            self._retained_capture_resources.append(
                _CaptureResourceSet(
                    pool=pool,
                    stream=capture_stream,
                    keepalives=[temporary],
                )
            )
            return

        for key, captured in temporary.items():
            self._reset_graph(captured.graph, context=f"capture rollback for {key}")
        self._retained_capture_resources.clear()
        temporary.clear()
        gc.collect()
        try:
            with torch.cuda.device(self._device):
                torch.cuda.empty_cache()
        except RuntimeError as cleanup_exc:
            logger.warning(
                "Qwen3-TTS incremental Codec graph rollback cleanup failed: %s",
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

    def decode(
        self,
        codes: torch.Tensor,
        state: Qwen3TTSIncrementalCodecState,
    ) -> IncrementalCodecGraphResult | None:
        """Replay the smallest captured bucket that fits this cohort."""

        if os.getpid() != self._owner_pid:
            raise RuntimeError(
                "Qwen3-TTS incremental Codec graph runner belongs to PID "
                f"{self._owner_pid}, but was used in PID {os.getpid()}"
            )
        if not self._enabled or not self._graphs:
            self._misses["disabled_or_uncaptured"] += 1
            return None
        self._validate_codes(codes)
        if int(codes.shape[2]) not in self._fresh_frames:
            self._misses["uncaptured_fresh_frames"] += 1
            return None

        batch_size = int(codes.shape[0])
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
            self._misses["missing_batch_bucket"] += 1
            return None

        key = IncrementalCodecGraphKey(int(codes.shape[2]), bucket)
        entry = self._graphs[key]
        entry.static_codes[:batch_size].copy_(codes)
        if batch_size < bucket:
            entry.static_codes[batch_size:].zero_()
        _copy_state_rows(entry.input_state, state, rows=batch_size)
        try:
            entry.graph.replay()
        except Exception as exc:
            self._replay_failures += 1
            reason = f"runtime_replay_failed: {type(exc).__name__}: {exc}"
            # Drop the last local graph reference before shared-pool cleanup.
            entry = None
            self._disable_runtime(reason)
            logger.exception(
                "Qwen3-TTS incremental Codec graph replay disabled the %s runner",
                self._mode,
            )
            raise
        self._replays += 1
        return IncrementalCodecGraphResult(
            waveform=entry.waveform[:batch_size],
            state=_slice_state_rows(entry.output_state, batch_size),
        )

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
        synchronized = False
        try:
            with torch.cuda.device(self._device):
                torch.cuda.synchronize(self._device)
            synchronized = True
        except RuntimeError as synchronize_exc:
            logger.warning(
                "Qwen3-TTS incremental Codec graph runtime synchronize failed; "
                "retaining graph buffers for the process lifetime: %s",
                synchronize_exc,
            )
        if not synchronized:
            return

        for key, captured in self._graphs.items():
            self._reset_graph(captured.graph, context=f"runtime disable for {key}")
        self._graphs.clear()
        self._pool = None
        self._capture_stream = None
        gc.collect()
        try:
            with torch.cuda.device(self._device):
                torch.cuda.empty_cache()
        except RuntimeError as cleanup_exc:
            logger.warning(
                "Qwen3-TTS incremental Codec graph runtime cleanup failed: %s",
                cleanup_exc,
            )

    def stats(self) -> dict[str, Any]:
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
                    for key in sorted(
                        self._graphs,
                        key=lambda item: (item.fresh_frames, item.batch_bucket),
                    )
                ],
            },
            "memory": dict(self._memory_stats),
            "retained_capture_resource_sets": len(self._retained_capture_resources),
            "runtime": {
                "replays": self._replays,
                "replay_failures": self._replay_failures,
                "fallback_counts": dict(sorted(self._misses.items())),
            },
        }


__all__ = [
    "IncrementalCodecGraphKey",
    "IncrementalCodecGraphResult",
    "Qwen3TTSIncrementalCodecCudaGraphRunner",
]
