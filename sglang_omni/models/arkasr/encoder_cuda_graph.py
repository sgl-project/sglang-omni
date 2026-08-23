# SPDX-License-Identifier: Apache-2.0
"""Bucketed CUDA graphs for the ARK-ASR audio encoder.

ARK-ASR's encoder path is the Whisper/RoPE tower plus the MLP frame-merge
adapter exposed as ``model.audio_encoder``.  Request-build pre-encoding can
batch mixed mel lengths after #1411, which makes the encoder a good CUDA graph
candidate: capture a padded, mask-aware bucket once, then replay the same full
encoder for later batches with the static input and mask buffers updated.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Sequence
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_BUCKETS = (1, 2, 4, 8)
_DEFAULT_FRAME_BUCKET_STEP = 256
_DEFAULT_MAX_FRAMES = 3000
_DEFAULT_MIN_FREE_GB = 3.0
_DEFAULT_WARMUP_ITERS = 3


def _normalize_batch_buckets(batch_buckets: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(bucket) for bucket in batch_buckets}))


def _bucket_batch(
    batch_size: int,
    batch_buckets: Sequence[int] = _DEFAULT_BATCH_BUCKETS,
) -> int | None:
    """Return the smallest configured batch bucket that can fit ``batch_size``."""
    for bucket in batch_buckets:
        if bucket >= batch_size:
            return int(bucket)
    return None


def _bucket_frames(
    mel_frames: int,
    *,
    frame_bucket_step: int = _DEFAULT_FRAME_BUCKET_STEP,
    max_frames: int = _DEFAULT_MAX_FRAMES,
) -> int | None:
    """Round ``mel_frames`` up to a graph bucket within the configured ceiling."""
    mel_frames = int(mel_frames)
    frame_bucket_step = int(frame_bucket_step)
    max_frames = int(max_frames)
    if mel_frames > max_frames:
        return None
    bucket = (
        (mel_frames + frame_bucket_step - 1) // frame_bucket_step
    ) * frame_bucket_step
    return min(max(bucket, frame_bucket_step), max_frames)


class ArkasrEncoderCudaGraphRunner:
    """Capture-once/replay-per-bucket runner for ARK-ASR encoder forwards.

    The runner owns one set of mutable static buffers per ``(batch, frames)``
    bucket.  Capture and replay are serialized because each replay overwrites
    those buffers; callers get a cloned tensor for the real batch rows before a
    later replay can reuse the bucket.
    """

    def __init__(
        self,
        audio_encoder: torch.nn.Module,
        *,
        batch_buckets: Iterable[int] = _DEFAULT_BATCH_BUCKETS,
        frame_bucket_step: int = _DEFAULT_FRAME_BUCKET_STEP,
        max_frames: int = _DEFAULT_MAX_FRAMES,
        min_free_gb: float = _DEFAULT_MIN_FREE_GB,
        warmup_iters: int = _DEFAULT_WARMUP_ITERS,
    ) -> None:
        self._audio_encoder = audio_encoder
        reference = next(audio_encoder.parameters())
        self._device = reference.device
        self._dtype = reference.dtype
        self._batch_buckets = _normalize_batch_buckets(batch_buckets)
        self._frame_bucket_step = int(frame_bucket_step)
        self._max_frames = int(max_frames)
        self._min_free_bytes = int(float(min_free_gb) * (1024**3))
        self._warmup_iters = int(warmup_iters)
        self._graphs: dict[tuple[int, int], tuple] = {}
        self._failed: set[tuple[int, int]] = set()
        self._pool = None
        self._lock = threading.Lock()
        self._done_event = torch.cuda.Event()
        self._event_recorded = False

    @property
    def batch_buckets(self) -> tuple[int, ...]:
        return self._batch_buckets

    def _enough_free_vram(self) -> tuple[bool, int]:
        if self._min_free_bytes <= 0:
            return True, 0
        free, _ = torch.cuda.mem_get_info(self._device)
        return free >= self._min_free_bytes, free

    def _forward(
        self,
        static_features: torch.Tensor,
        static_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._audio_encoder(static_features, attention_mask=static_mask)

    def _capture(
        self,
        batch_bucket: int,
        frame_bucket: int,
        mel_bins: int,
    ) -> tuple:
        static_features = torch.zeros(
            batch_bucket,
            mel_bins,
            frame_bucket,
            device=self._device,
            dtype=self._dtype,
        )
        static_mask = torch.zeros(
            batch_bucket,
            frame_bucket,
            device=self._device,
            dtype=torch.bool,
        )
        # Padded rows keep one valid silent frame so every row stays finite
        # through attention. Real rows overwrite their masks before replay.
        static_mask[:, 0] = True

        stream = torch.cuda.Stream(device=self._device)
        stream.wait_stream(torch.cuda.current_stream(self._device))
        with torch.cuda.stream(stream):
            for _ in range(self._warmup_iters):
                self._forward(static_features, static_mask)
        torch.cuda.current_stream(self._device).wait_stream(stream)
        torch.cuda.synchronize(self._device)

        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            pool=self._pool,
            capture_error_mode="thread_local",
        ):
            static_out = self._forward(static_features, static_mask)
        frame_index = torch.arange(frame_bucket, device=self._device).unsqueeze(0)

        logger.info(
            "Captured ARK-ASR encoder CUDA graph batch=%d frames=%d -> out %s "
            "(%d cached)",
            batch_bucket,
            frame_bucket,
            tuple(static_out.shape),
            len(self._graphs) + 1,
        )
        return graph, static_features, static_mask, frame_index, static_out

    def _copy_inputs(
        self,
        static_features: torch.Tensor,
        static_mask: torch.Tensor,
        static_frame_index: torch.Tensor,
        features: torch.Tensor,
        lengths: list[int],
    ) -> None:
        batch_size, _mel_bins, mel_frames = features.shape
        frame_bucket = static_features.shape[-1]
        static_features.zero_()
        static_features[:batch_size, :, :mel_frames].copy_(
            features,
            non_blocking=True,
        )
        static_mask.zero_()
        # Padded batch rows use one valid zero frame and are discarded.
        static_mask[:, 0] = True
        real_lengths = torch.as_tensor(lengths, device=self._device, dtype=torch.long)
        static_mask[:batch_size].copy_(
            static_frame_index[:, :frame_bucket] < real_lengths.unsqueeze(1),
            non_blocking=True,
        )

    @torch.no_grad()
    def run(
        self,
        features: torch.Tensor,
        mel_lengths: Sequence[int],
    ) -> Optional[torch.Tensor]:
        """Replay a graph for ``features`` or return ``None`` for eager fallback.

        Args:
            features: Padded mel batch ``[B, mel_bins, T]`` already on the
                encoder device/dtype.
            mel_lengths: Valid mel-frame count for each real batch row.

        Returns:
            ``[B, T_bucket', hidden]`` for real rows, cloned away from the
            static graph output buffer, or ``None`` when the caller should use
            the eager encoder path.
        """
        batch_size, mel_bins, mel_frames = features.shape
        lengths = [int(length) for length in mel_lengths]

        batch_bucket = _bucket_batch(batch_size, self._batch_buckets)
        frame_bucket = _bucket_frames(
            mel_frames,
            frame_bucket_step=self._frame_bucket_step,
            max_frames=self._max_frames,
        )
        if batch_bucket is None or frame_bucket is None:
            return None
        key = (batch_bucket, frame_bucket)
        if key in self._failed:
            return None

        with self._lock:
            # A queued caller may have observed this bucket before the caller
            # holding the lock marked it failed.
            if key in self._failed:
                return None

            entry = self._graphs.get(key)
            if entry is None:
                enough, free = self._enough_free_vram()
                if not enough:
                    logger.warning(
                        "ARK-ASR encoder CUDA graph: free VRAM %.1fGB < %.1fGB "
                        "headroom; running batch=%d frames=%d eager",
                        free / 1024**3,
                        self._min_free_bytes / 1024**3,
                        batch_bucket,
                        frame_bucket,
                    )
                    self._failed.add(key)
                    return None
                try:
                    with torch.cuda.device(self._device):
                        entry = self._capture(batch_bucket, frame_bucket, mel_bins)
                except Exception as exc:
                    logger.warning(
                        "ARK-ASR encoder CUDA graph capture failed for "
                        "batch=%d frames=%d: %s; using eager for this bucket",
                        batch_bucket,
                        frame_bucket,
                        exc,
                    )
                    self._failed.add(key)
                    return None
                self._graphs[key] = entry

            graph, static_features, static_mask, static_frame_index, static_out = entry
            stream = torch.cuda.current_stream(self._device)
            if self._event_recorded:
                self._done_event.wait(stream)
            try:
                self._copy_inputs(
                    static_features,
                    static_mask,
                    static_frame_index,
                    features,
                    lengths,
                )
                graph.replay()
                out = static_out[:batch_size].clone()
                self._done_event.record(stream)
                self._event_recorded = True
                return out
            except Exception as exc:
                logger.warning(
                    "ARK-ASR encoder CUDA graph replay failed for "
                    "batch=%d frames=%d: %s; using eager for this bucket",
                    batch_bucket,
                    frame_bucket,
                    exc,
                )
                self._failed.add(key)
                return None


__all__ = [
    "ArkasrEncoderCudaGraphRunner",
    "_bucket_batch",
    "_bucket_frames",
]
