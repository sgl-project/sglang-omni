# SPDX-License-Identifier: Apache-2.0
"""Bucketed CUDA graphs for the ARK-ASR audio encoder.

ARK-ASR's encoder path is the Whisper/RoPE tower plus the MLP frame-merge
adapter exposed as ``model.audio_encoder``.  Request-build pre-encoding can
batch mixed mel lengths after #1411, which makes the encoder a good CUDA graph
candidate: capture a padded, mask-aware bucket once, then replay the same full
encoder for later batches with the static input and length buffers updated.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_BUCKETS = (1, 2, 4, 8)
_DEFAULT_FRAME_BUCKETS = (512, 768, 1024)
_DEFAULT_MIN_FREE_GB = 3.0


@dataclass
class _CapturedGraph:
    graph: torch.cuda.CUDAGraph
    features: torch.Tensor
    lengths: torch.Tensor
    output: torch.Tensor


def _bucket_batch(batch_size: int) -> int | None:
    """Return the smallest captured batch bucket that fits ``batch_size``."""
    for bucket in _DEFAULT_BATCH_BUCKETS:
        if bucket >= batch_size:
            return bucket
    return None


def _bucket_frames(mel_frames: int) -> int | None:
    """Return the smallest captured frame bucket that fits ``mel_frames``."""
    for bucket in _DEFAULT_FRAME_BUCKETS:
        if bucket >= mel_frames:
            return bucket
    return None


class ArkasrEncoderCudaGraphRunner:
    """Startup-captured CUDA graph runner for ARK-ASR encoder forwards.

    The runner owns one set of mutable static buffers per ``(batch, frames)``
    bucket. Replay is serialized because each replay overwrites those buffers;
    callers get a cloned tensor for the real batch rows before a later replay
    can reuse the bucket. Shapes not captured before serving use eager fallback.
    """

    def __init__(
        self,
        audio_encoder: torch.nn.Module,
        *,
        min_free_gb: float = _DEFAULT_MIN_FREE_GB,
    ) -> None:
        self._audio_encoder = audio_encoder
        reference = next(audio_encoder.parameters())
        self._device = reference.device
        self._dtype = reference.dtype
        self._mel_bins = int(audio_encoder.whisper.conv1.in_channels)
        self._min_free_bytes = int(float(min_free_gb) * (1024**3))
        self._graphs: dict[tuple[int, int], _CapturedGraph] = {}
        self._pool = None
        self._lock = threading.Lock()
        self._done_event = torch.cuda.Event() if self._device.type == "cuda" else None
        self._event_recorded = False

    def _enough_free_vram(self) -> tuple[bool, int]:
        if self._min_free_bytes <= 0:
            return True, 0
        free, _ = torch.cuda.mem_get_info(self._device)
        return free >= self._min_free_bytes, free

    def _capture(
        self,
        batch_bucket: int,
        frame_bucket: int,
    ) -> _CapturedGraph:
        static_features = torch.zeros(
            batch_bucket,
            self._mel_bins,
            frame_bucket,
            device=self._device,
            dtype=self._dtype,
        )
        static_lengths = torch.ones(
            batch_bucket,
            device=self._device,
            dtype=torch.long,
        )

        def _masked_forward() -> torch.Tensor:
            frame_index = torch.arange(frame_bucket, device=self._device).unsqueeze(0)
            mask = frame_index < static_lengths.unsqueeze(1)
            return self._audio_encoder(static_features, attention_mask=mask)

        stream = torch.cuda.Stream(device=self._device)
        stream.wait_stream(torch.cuda.current_stream(self._device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                _masked_forward()
        torch.cuda.current_stream(self._device).wait_stream(stream)
        torch.cuda.synchronize(self._device)

        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            static_out = _masked_forward()

        logger.info(
            "Captured ARK-ASR encoder CUDA graph batch=%d frames=%d -> out %s "
            "(%d cached)",
            batch_bucket,
            frame_bucket,
            tuple(static_out.shape),
            len(self._graphs) + 1,
        )
        return _CapturedGraph(
            graph=graph,
            features=static_features,
            lengths=static_lengths,
            output=static_out,
        )

    @torch.no_grad()
    def capture_startup_buckets(self) -> None:
        """Capture the supported serving shapes before the server becomes ready."""
        if self._device.type != "cuda":
            return
        with self._lock:
            for batch_bucket in reversed(_DEFAULT_BATCH_BUCKETS):
                for frame_bucket in reversed(_DEFAULT_FRAME_BUCKETS):
                    enough, free = self._enough_free_vram()
                    if not enough:
                        logger.warning(
                            "ARK-ASR encoder CUDA graph: free VRAM %.1fGB < %.1fGB "
                            "headroom; skipping batch=%d frames=%d",
                            free / 1024**3,
                            self._min_free_bytes / 1024**3,
                            batch_bucket,
                            frame_bucket,
                        )
                        continue
                    try:
                        with torch.cuda.device(self._device):
                            self._graphs[(batch_bucket, frame_bucket)] = self._capture(
                                batch_bucket,
                                frame_bucket,
                            )
                    except Exception as exc:
                        logger.warning(
                            "ARK-ASR encoder CUDA graph capture failed for "
                            "batch=%d frames=%d: %s; using eager for this bucket",
                            batch_bucket,
                            frame_bucket,
                            exc,
                        )

    def _copy_inputs(
        self,
        entry: _CapturedGraph,
        features: torch.Tensor,
        lengths: Sequence[int],
    ) -> None:
        batch_size, _, mel_frames = features.shape
        entry.features.zero_()
        entry.features[:batch_size, :, :mel_frames].copy_(features)
        # Padded rows keep one valid zero frame and are discarded.
        entry.lengths.fill_(1)
        entry.lengths[:batch_size].copy_(
            torch.as_tensor(lengths, dtype=torch.long),
            non_blocking=True,
        )

    @torch.no_grad()
    def run(
        self,
        features: torch.Tensor,
        mel_lengths: Sequence[int],
    ) -> torch.Tensor | None:
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
        batch_size, _, mel_frames = features.shape

        batch_bucket = _bucket_batch(batch_size)
        frame_bucket = _bucket_frames(mel_frames)
        if batch_bucket is None or frame_bucket is None:
            return None
        key = (batch_bucket, frame_bucket)

        with self._lock:
            entry = self._graphs.get(key)
            if entry is None:
                return None

            stream = torch.cuda.current_stream(self._device)
            if self._event_recorded:
                assert self._done_event is not None
                self._done_event.wait(stream)
            self._copy_inputs(entry, features, mel_lengths)
            try:
                entry.graph.replay()
            except Exception as exc:
                logger.warning(
                    "ARK-ASR encoder CUDA graph replay failed for "
                    "batch=%d frames=%d: %s; using eager for this bucket",
                    batch_bucket,
                    frame_bucket,
                    exc,
                )
                self._graphs.pop(key, None)
                return None
            out = entry.output[:batch_size].clone()
            assert self._done_event is not None
            self._done_event.record(stream)
            self._event_recorded = True
            return out


__all__ = [
    "ArkasrEncoderCudaGraphRunner",
]
