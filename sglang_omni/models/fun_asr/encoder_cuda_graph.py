# SPDX-License-Identifier: Apache-2.0
"""Bucketed CUDA graphs for the Fun-ASR audio encoder + adaptor.

The encoder forward is launch-bound: hundreds of small kernels whose CPU-side
dispatch (Python launcher glue + driver calls) dwarfs the GPU time. Replaying
a captured graph reduces the whole forward to a single launch.

Unlike the MOSS-TD Whisper runner (fixed input length, only the chunk count
varies), Fun-ASR varies in two dims: batch size (1..pre_lm_max_batch_size)
and LFR frame count (up to ~500 for the 30 s clip limit). We bucket both and
pad up on replay:

* batch rows are padded with ``ilens=1`` silence rows (a fully-masked row
  would produce NaN through SDPA; one valid zero frame keeps the row finite
  and its output is discarded),
* time is padded with masked frames — the same masking the eager batched
  path already applies to every non-longest item in a batch.

The SANM mask is derived from a static lengths tensor *inside* the capture,
so a replay only needs ``copy_`` of the input features and lengths.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional, Tuple

import torch

from .sglang_model import _sanm_mask_from_lengths

logger = logging.getLogger(__name__)

_BATCH_BUCKETS = (1, 2, 4, 8)
_T_BUCKET_STEP = 64
_T_BUCKET_MAX = 512  # 30 s * (1000 ms / 60 ms per LFR frame) ~= 500 frames


def _bucket_batch(b: int, max_batch: int) -> int | None:
    for bucket in _BATCH_BUCKETS:
        if bucket > max_batch:
            break
        if bucket >= b:
            return bucket
    return max_batch if b <= max_batch else None


def _bucket_t(t: int) -> int | None:
    if t > _T_BUCKET_MAX:
        return None
    bucket = ((t + _T_BUCKET_STEP - 1) // _T_BUCKET_STEP) * _T_BUCKET_STEP
    return max(bucket, _T_BUCKET_STEP)


class FunASREncoderCudaGraphRunner:
    """Capture-once/replay per (batch, LFR-length) bucket.

    Holds references to the *eager* audio_tower and multi_modal_projector;
    capturing dynamo-compiled callables is unsupported.
    """

    def __init__(
        self,
        audio_tower,
        multi_modal_projector,
        *,
        max_batch_size: int = 8,
        min_free_gb: float = 3.0,
        warmup_iters: int = 3,
    ) -> None:
        self._audio_tower = audio_tower
        self._projector = multi_modal_projector
        reference = next(audio_tower.parameters())
        self._device = reference.device
        self._dtype = reference.dtype
        self._max_batch = max(int(max_batch_size), 1)
        self._min_free_bytes = int(float(min_free_gb) * (1024**3))
        self._warmup_iters = int(warmup_iters)
        # (batch_bucket, t_bucket) -> (graph, static_xs, static_ilens, static_out)
        self._graphs: dict[Tuple[int, int], tuple] = {}
        self._failed: set[Tuple[int, int]] = set()
        self._pool = None
        # note (wilsonzheng0327): serializes capture and replay -- replay
        # mutates the bucket's static buffers, and both the pre-LM worker and
        # the scheduler's inline prefill path can reach get_audio_feature.
        self._lock = threading.Lock()
        self._done_event = torch.cuda.Event()
        self._event_recorded = False

    def _forward(self, xs: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        enc_out = self._audio_tower(xs, mask)
        return self._projector(enc_out, mask)

    def _enough_free_vram(self) -> tuple[bool, int]:
        free, _ = torch.cuda.mem_get_info(self._device)
        return free >= self._min_free_bytes, free

    def _capture(self, batch_bucket: int, t_bucket: int, feat_dim: int) -> tuple:
        static_xs = torch.zeros(
            batch_bucket, t_bucket, feat_dim, device=self._device, dtype=self._dtype
        )
        static_ilens = torch.ones(batch_bucket, device=self._device, dtype=torch.long)

        def _masked_forward() -> torch.Tensor:
            mask = _sanm_mask_from_lengths(
                static_ilens, t_bucket, dtype=self._dtype, device=self._device
            )
            return self._forward(static_xs, mask)

        # note (wilsonzheng0327): warmup on a fresh stream so allocator state
        # settles before capture.
        stream = torch.cuda.Stream(device=self._device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self._warmup_iters):
                _masked_forward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        # note (wilsonzheng0327): thread_local error mode -- the LM scheduler
        # thread keeps launching kernels concurrently and must not poison this
        # thread's capture.
        with torch.cuda.graph(
            graph, pool=self._pool, capture_error_mode="thread_local"
        ):
            static_out = _masked_forward()
        logger.info(
            "Captured Fun-ASR encoder CUDA graph batch=%d t=%d -> out %s "
            "(%d cached)",
            batch_bucket,
            t_bucket,
            tuple(static_out.shape),
            len(self._graphs) + 1,
        )
        return graph, static_xs, static_ilens, static_out

    @torch.no_grad()
    def run(self, xs: torch.Tensor, lengths: List[int]) -> Optional[torch.Tensor]:
        """Replay for ``xs`` [B, T, feat] with per-item valid ``lengths``.

        Returns adaptor output ``[B, T', llm_dim]`` for the real batch rows,
        or None when no bucket fits / capture failed (caller falls back to
        the eager path).
        """
        b, t, feat_dim = xs.shape
        batch_bucket = _bucket_batch(b, self._max_batch)
        t_bucket = _bucket_t(t)
        if batch_bucket is None or t_bucket is None:
            return None
        key = (batch_bucket, t_bucket)
        if key in self._failed:
            return None

        with self._lock:
            entry = self._graphs.get(key)
            if entry is None:
                enough, free = self._enough_free_vram()
                if not enough:
                    logger.warning(
                        "Fun-ASR encoder CUDA graph: free VRAM %.1fGB < %.1fGB "
                        "headroom; running batch=%d t=%d eager",
                        free / 1024**3,
                        self._min_free_bytes / 1024**3,
                        batch_bucket,
                        t_bucket,
                    )
                    self._failed.add(key)
                    return None
                try:
                    with torch.cuda.device(self._device):
                        entry = self._capture(batch_bucket, t_bucket, feat_dim)
                except Exception as exc:
                    logger.warning(
                        "Fun-ASR encoder CUDA graph capture failed for "
                        "batch=%d t=%d: %s; using eager for this bucket",
                        batch_bucket,
                        t_bucket,
                        exc,
                    )
                    self._failed.add(key)
                    return None
                self._graphs[key] = entry

            graph, static_xs, static_ilens, static_out = entry
            if static_xs.shape[-1] != feat_dim:
                return None
            stream = torch.cuda.current_stream(self._device)
            # note (wilsonzheng0327): wait for previous caller's output copy
            # on some stream to finish before using shared resource
            if self._event_recorded:
                self._done_event.wait(stream)
            static_xs.zero_()
            static_xs[:b, :t].copy_(xs, non_blocking=True)
            # Padded rows keep ilens=1: one valid zeroed frame, output dropped.
            static_ilens.fill_(1)
            static_ilens[:b].copy_(
                torch.as_tensor(lengths, dtype=torch.long), non_blocking=True
            )
            graph.replay()
            # note (wilsonzheng0327): the next call needs to wait on this
            # event before it touches anything shared to ensure clone finishes
            out = static_out[:b].clone()
            self._done_event.record(stream)
            self._event_recorded = True
            return out


__all__ = ["FunASREncoderCudaGraphRunner"]
