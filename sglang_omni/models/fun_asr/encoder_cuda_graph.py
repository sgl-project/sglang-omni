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
from typing import Any, List, Optional, Tuple

import torch
from sglang.srt.utils.common import get_available_gpu_memory

from sglang_omni.platforms import current_platform

from .sglang_model import _sanm_mask_from_lengths

logger = logging.getLogger(__name__)

# Device types whose platform graph backend records work issued through the same
# module this runner drives for streams, events and memory. MUSA is left out on
# purpose: its platform supplies the CUDA backend, which captures through
# torch.cuda, while a musa tensor resolves to torch.musa, so warmup and capture
# would straddle two device surfaces. ROCm belongs here because it genuinely
# presents as torch.cuda.
_SAME_SURFACE_DEVICE_TYPES = frozenset({"cuda", "xpu"})

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
        device_module: Any | None = None,
        graph_device_types: frozenset[str] | None = None,
    ) -> None:
        self._audio_tower = audio_tower
        self._projector = multi_modal_projector
        reference = next(audio_tower.parameters())
        self._device = reference.device
        self._dtype = reference.dtype
        self._device_module = device_module or torch.get_device_module(self._device)
        self._graph_backend = current_platform.get_device_graph_backend(self._device)
        same_surface = (
            _SAME_SURFACE_DEVICE_TYPES
            if graph_device_types is None
            else graph_device_types
        )
        if self._graph_backend is not None and self._device.type not in same_surface:
            logger.info(
                "Fun-ASR encoder graphs stay off on %s: its capture backend and "
                "its device module are different surfaces",
                self._device.type,
            )
            self._graph_backend = None
        if self._graph_backend is None:
            logger.info(
                "Fun-ASR encoder graphs are unavailable on %s; the encoder runs "
                "eager",
                self._device,
            )
        self._max_batch = max(int(max_batch_size), 1)
        self._min_free_gb = float(min_free_gb)
        self._warmup_iters = int(warmup_iters)
        # (batch_bucket, t_bucket) -> (graph, static_xs, static_ilens, static_out)
        self._graphs: dict[Tuple[int, int], tuple] = {}
        self._failed: set[Tuple[int, int]] = set()
        # note (wilsonzheng0327): serializes capture and replay -- replay
        # mutates the bucket's static buffers, and both the pre-LM worker and
        # the scheduler's inline prefill path can reach get_audio_feature.
        self._lock = threading.Lock()
        # A capture may not run while another capture is underway, but a replay
        # may: captures take seconds, so they wait on their own lock instead of
        # the one that serializes replays.
        self._capture_lock = threading.Lock()
        # Only the graph path uses this, and a device module that records no graph
        # may not offer an event either: torch.mps.Event() raises without the MPS
        # backend, which would fail stage setup instead of falling back to eager.
        self._done_event = (
            self._device_module.Event() if self._graph_backend is not None else None
        )
        self._event_recorded = False

    def _forward(self, xs: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        enc_out = self._audio_tower(xs, mask)
        return self._projector(enc_out, mask)

    def _enough_free_gb(self) -> tuple[bool, float]:
        # Both flags are pinned rather than defaulted: a capture is triggered
        # lazily by whichever worker sees a new bucket first, so this must never
        # become a collective, and dropping the allocator's cached blocks here
        # would both stall and take away what the graph pool reuses.
        free_gb = get_available_gpu_memory(
            self._device.type,
            self._device.index or 0,
            distributed=False,
            empty_cache=False,
        )
        return free_gb >= self._min_free_gb, free_gb

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

        with current_platform.graph_capture_attention():
            # note (wilsonzheng0327): warmup on a fresh stream so allocator state
            # settles before capture.
            stream = self._device_module.Stream(device=self._device)
            stream.wait_stream(self._device_module.current_stream())
            with self._device_module.stream(stream):
                for _ in range(self._warmup_iters):
                    _masked_forward()
            self._device_module.current_stream().wait_stream(stream)
            self._device_module.synchronize()

            # Each bucket keeps its own pool: memory in a shared pool is reused
            # by the next graph recorded into it, which is safe only while
            # replays follow capture order. A request picks its bucket from the
            # clip length, and captures deliberately run outside the replay lock,
            # so a replay of one bucket can overlap the capture of another. Same
            # rule as the Qwen3-ASR encoder runner. The cost is a private pool
            # per bucket -- measured on a B60 at 24 to 336 MiB, largest bucket
            # dominating -- bounded by the headroom check above, which stops
            # capturing and leaves later buckets eager rather than exhausting the
            # card.
            with self._graph_backend.capture(thread_local_errors=True) as graph:
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

    def _capture_bucket(self, key: Tuple[int, int], feat_dim: int) -> Optional[tuple]:
        """Capture one bucket, or None if it cannot be captured.

        Held under the capture lock rather than the replay lock, so a capture
        does not stall encodes for the buckets that are already recorded.
        """
        batch_bucket, t_bucket = key
        with self._capture_lock:
            with self._lock:
                entry = self._graphs.get(key)
                declined = key in self._failed
            if entry is not None:
                return entry
            if declined:
                return None

            enough, free_gb = self._enough_free_gb()
            if not enough:
                logger.warning(
                    "Fun-ASR encoder CUDA graph: free VRAM %.1fGB < %.1fGB "
                    "headroom; running batch=%d t=%d eager",
                    free_gb,
                    self._min_free_gb,
                    batch_bucket,
                    t_bucket,
                )
                with self._lock:
                    self._failed.add(key)
                return None

            try:
                with self._device_module.device(self._device):
                    entry = self._capture(batch_bucket, t_bucket, feat_dim)
            except Exception as exc:
                logger.warning(
                    "Fun-ASR encoder CUDA graph capture failed for "
                    "batch=%d t=%d: %s; using eager for this bucket",
                    batch_bucket,
                    t_bucket,
                    exc,
                    exc_info=True,
                )
                with self._lock:
                    self._failed.add(key)
                return None

            with self._lock:
                self._graphs[key] = entry
            return entry

    @torch.no_grad()
    def run(self, xs: torch.Tensor, lengths: List[int]) -> Optional[torch.Tensor]:
        """Replay for ``xs`` [B, T, feat] with per-item valid ``lengths``.

        Returns adaptor output ``[B, T', llm_dim]`` for the real batch rows,
        or None when no bucket fits / capture failed (caller falls back to
        the eager path).
        """
        if self._graph_backend is None:
            return None
        b, t, feat_dim = xs.shape
        batch_bucket = _bucket_batch(b, self._max_batch)
        t_bucket = _bucket_t(t)
        if batch_bucket is None or t_bucket is None:
            return None
        key = (batch_bucket, t_bucket)

        with self._lock:
            entry = self._graphs.get(key)
            declined = key in self._failed
        if entry is None:
            if declined:
                return None
            entry = self._capture_bucket(key, feat_dim)
            if entry is None:
                return None

        with self._lock:
            graph, static_xs, static_ilens, static_out = entry
            if static_xs.shape[-1] != feat_dim:
                return None
            stream = self._device_module.current_stream(self._device)
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
