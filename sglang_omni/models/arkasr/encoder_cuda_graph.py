# SPDX-License-Identifier: Apache-2.0
"""Bucketed CUDA graphs for the ARK-ASR audio encoder + MLP adapter.

Batch size and mel-frame count both vary, so we bucket both and pad on
replay. Startup precaptures the short-clip working set; other shapes stay
eager, and requests never trigger capture.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import torch
from torch import nn

logger = logging.getLogger(__name__)

_T_BUCKET_STEP = 64
# note (guozhihao-224): ~10 s working set. Full 30 s is ~180 graphs and stays eager.
_PRECAPTURE_MEL_FRAMES = 1024


@dataclass
class _CapturedGraph:
    graph: torch.cuda.CUDAGraph
    static_mel: torch.Tensor
    static_ilens: torch.Tensor
    static_out: torch.Tensor


def _t_step(merge_factor: int) -> int:
    align = 2 * max(merge_factor, 1)
    step = max(_T_BUCKET_STEP, align)
    return (step // align) * align


def _batch_buckets(max_batch: int) -> tuple[int, ...]:
    """Powers of two up to max_batch, then max_batch itself."""
    limit = max(max_batch, 1)
    buckets: list[int] = []
    size = 1
    while size < limit:
        buckets.append(size)
        size *= 2
    buckets.append(limit)
    return tuple(buckets)


def _t_buckets_upto(
    max_t: int,
    *,
    merge_factor: int,
    max_mel_frames: int | None = None,
) -> tuple[int, ...]:
    """Inclusive T buckets from the step size up to max_t."""
    step = _t_step(merge_factor)
    limit = max_t
    if max_mel_frames is not None:
        limit = min(limit, max_mel_frames)
    buckets: list[int] = []
    t = step
    while t <= limit:
        buckets.append(t)
        t += step
    return tuple(buckets)


def _fit_bucket(value: int, buckets: tuple[int, ...]) -> int | None:
    """Smallest captured bucket that fits value, or None."""
    if value < 1:
        return None
    for bucket in buckets:
        if bucket >= value:
            return bucket
    return None


class ArkasrEncoderCudaGraphRunner:
    """Startup capture / replay per (batch, mel-length) bucket.

    Holds a reference to the eager ArkAudioMLPAdapter; capturing a
    dynamo-compiled callable is unsupported. run() never captures.
    """

    def __init__(
        self,
        audio_encoder: nn.Module,
        *,
        max_batch_size: int = 8,
        max_mel_frames: int | None = None,
        merge_factor: int | None = None,
        min_free_gb: float = 3.0,
    ) -> None:
        self._audio_encoder = audio_encoder
        reference = next(audio_encoder.parameters())
        self._device = reference.device
        self._dtype = reference.dtype
        self._max_batch = max(max_batch_size, 1)
        self._batch_buckets = _batch_buckets(self._max_batch)
        self._max_mel_frames = (
            max(max_mel_frames, 1) if max_mel_frames is not None else None
        )
        self._merge_factor = max(
            audio_encoder.merge_factor if merge_factor is None else merge_factor,
            1,
        )
        self._t_buckets: tuple[int, ...] = ()
        self._min_free_bytes = int(float(min_free_gb) * (1024**3))
        self._graphs: dict[tuple[int, int], _CapturedGraph] = {}
        self._failed: set[tuple[int, int]] = set()
        self._pool = None
        self._logged_replay_buckets: set[tuple[int, int]] = set()
        # note (guozhihao-224): replay mutates the bucket's static buffers, and
        # both the pre-LM worker and the scheduler's inline prefill path can
        # reach get_audio_feature.
        self._lock = threading.Lock()
        self._done_event = torch.cuda.Event() if self._device.type == "cuda" else None
        self._event_recorded = False

    @property
    def captured_buckets(self) -> tuple[tuple[int, int], ...]:
        """Return captured (batch, T) buckets in ascending order."""
        return tuple(sorted(self._graphs))

    def _mel_mask(self, ilens: torch.Tensor, t_bucket: int) -> torch.Tensor:
        frame_index = torch.arange(t_bucket, device=self._device).unsqueeze(0)
        return frame_index < ilens.unsqueeze(1)

    def _forward(self, mel: torch.Tensor, ilens: torch.Tensor) -> torch.Tensor:
        mask = self._mel_mask(ilens, mel.shape[-1])
        return self._audio_encoder(mel, attention_mask=mask)

    def _enough_free_vram(self) -> tuple[bool, int]:
        free, _ = torch.cuda.mem_get_info(self._device)
        return free >= self._min_free_bytes, free

    def _capture(
        self, batch_bucket: int, t_bucket: int, num_mel_bins: int
    ) -> _CapturedGraph:
        static_mel = torch.zeros(
            batch_bucket,
            num_mel_bins,
            t_bucket,
            device=self._device,
            dtype=self._dtype,
        )
        static_ilens = torch.ones(batch_bucket, device=self._device, dtype=torch.long)

        def _masked_forward() -> torch.Tensor:
            return self._forward(static_mel, static_ilens)

        stream = torch.cuda.Stream(device=self._device)
        stream.wait_stream(torch.cuda.current_stream(self._device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                _masked_forward()
        # note (guozhihao-224): sync only the warmup stream. A device-wide
        # synchronize would freeze any concurrent CUDA work on this device.
        stream.synchronize()

        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        # note (guozhihao-224): thread_local isolates capture from other CUDA
        # threads that may still exist after generation-graph warmup.
        with torch.cuda.graph(
            graph, pool=self._pool, capture_error_mode="thread_local"
        ):
            static_out = _masked_forward()
        logger.info(
            "Captured ARK-ASR encoder CUDA graph batch=%d t=%d -> out %s "
            "(%d cached)",
            batch_bucket,
            t_bucket,
            tuple(static_out.shape),
            len(self._graphs) + 1,
        )
        return _CapturedGraph(graph, static_mel, static_ilens, static_out)

    def _capture_bucket(
        self, batch_bucket: int, t_bucket: int, num_mel_bins: int
    ) -> _CapturedGraph | None:
        """Capture one startup bucket. Caller holds _lock."""
        key = (batch_bucket, t_bucket)
        if key in self._failed or key in self._graphs:
            return self._graphs.get(key)
        enough, free = self._enough_free_vram()
        if not enough:
            logger.warning(
                "ARK-ASR encoder CUDA graph: free VRAM %.1fGB < %.1fGB "
                "headroom; skipping batch=%d t=%d",
                free / 1024**3,
                self._min_free_bytes / 1024**3,
                batch_bucket,
                t_bucket,
            )
            self._failed.add(key)
            return None
        try:
            with torch.cuda.device(self._device):
                entry = self._capture(batch_bucket, t_bucket, num_mel_bins)
        except Exception as exc:
            logger.warning(
                "ARK-ASR encoder CUDA graph capture failed for "
                "batch=%d t=%d: %s; using eager for this bucket",
                batch_bucket,
                t_bucket,
                exc,
            )
            self._failed.add(key)
            return None
        self._graphs[key] = entry
        return entry

    @torch.no_grad()
    def capture_working_set(
        self,
        num_mel_bins: int,
        *,
        max_mel_frames: int | None = None,
    ) -> None:
        """Capture short-clip (batch, T) buckets before serving traffic.

        max_mel_frames defaults to _PRECAPTURE_MEL_FRAMES. Uncaptured shapes
        stay on the eager path; run() never captures.
        """
        if self._device.type != "cuda":
            return
        t_limit = (
            max_mel_frames if max_mel_frames is not None else _PRECAPTURE_MEL_FRAMES
        )
        t_buckets = _t_buckets_upto(
            t_limit,
            merge_factor=self._merge_factor,
            max_mel_frames=self._max_mel_frames,
        )
        batch_buckets = self._batch_buckets
        # note (guozhihao-224): largest-first so the shared CUDA graph pool
        # is sized by the biggest capture. Capturing a larger graph later
        # can grow the pool and invalidate earlier graphs.
        keys = [(b, t) for t in reversed(t_buckets) for b in reversed(batch_buckets)]
        logger.info(
            "ARK-ASR encoder CUDA graph precapture %d buckets "
            "(batch=%s, t<=%d, num_mel_bins=%d)",
            len(keys),
            batch_buckets,
            t_limit,
            num_mel_bins,
        )
        with self._lock:
            self._t_buckets = t_buckets
            for batch_bucket, t_bucket in keys:
                self._capture_bucket(batch_bucket, t_bucket, num_mel_bins)
        logger.info(
            "ARK-ASR encoder CUDA graph precapture done (%d cached, %d failed)",
            len(self._graphs),
            len(self._failed),
        )

    @torch.no_grad()
    def run(self, mel: torch.Tensor, lengths: list[int]) -> torch.Tensor | None:
        """Replay for mel [B, n_mels, T] with per-item valid lengths.

        Returns adapter output [B, T', llm_dim] for the real batch rows,
        or None when the shape was not captured at startup / replay failed
        (caller falls back to the eager path). Requests never trigger capture.
        """
        if self._device.type != "cuda" or mel.ndim != 3:
            return None
        b, num_mel_bins, t = mel.shape
        batch_bucket = _fit_bucket(b, self._batch_buckets)
        t_bucket = _fit_bucket(t, self._t_buckets)
        if batch_bucket is None or t_bucket is None:
            return None
        key = (batch_bucket, t_bucket)
        with self._lock:
            if key in self._failed:
                return None
            entry = self._graphs.get(key)
            if entry is None:
                return None
            if entry.static_mel.shape[1] != num_mel_bins:
                return None
            stream = torch.cuda.current_stream(self._device)
            if self._event_recorded and self._done_event is not None:
                self._done_event.wait(stream)
            entry.static_mel.zero_()
            entry.static_mel[:b, :, :t].copy_(mel, non_blocking=True)
            # note (guozhihao-224): padded rows keep ilens=1 (one valid zeroed
            # frame); the extra output rows are dropped below.
            entry.static_ilens.fill_(1)
            entry.static_ilens[:b].copy_(
                torch.as_tensor(lengths, dtype=torch.long, device=self._device),
                non_blocking=True,
            )
            try:
                entry.graph.replay()
            except Exception as exc:
                logger.warning(
                    "ARK-ASR encoder CUDA graph replay failed for "
                    "batch=%d t=%d: %s; using eager for this bucket",
                    batch_bucket,
                    t_bucket,
                    exc,
                )
                self._graphs.pop(key, None)
                self._failed.add(key)
                return None
            if key not in self._logged_replay_buckets:
                logger.info(
                    "Replaying ARK-ASR encoder CUDA graph batch=%d t=%d "
                    "request_batch=%d request_t=%d",
                    batch_bucket,
                    t_bucket,
                    b,
                    t,
                )
                self._logged_replay_buckets.add(key)
            out = entry.static_out[:b].clone()
            if self._done_event is not None:
                self._done_event.record(stream)
                self._event_recorded = True
            return out


__all__ = ["ArkasrEncoderCudaGraphRunner"]
