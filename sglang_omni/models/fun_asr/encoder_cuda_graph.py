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

Streams, events and pools reach the device through torch.get_device_module, so
the only per-device knowledge left is the graph type and the capture arguments,
which _graph_api spells out.
"""

from __future__ import annotations

import logging
import threading
from contextlib import AbstractContextManager, nullcontext
from typing import Any, List, Optional, Tuple

import torch
from torch.nn.attention import sdpa_kernel

from sglang_omni.platforms import current_platform

from .sglang_model import _sanm_mask_from_lengths

logger = logging.getLogger(__name__)

_BATCH_BUCKETS = (1, 2, 4, 8)
_T_BUCKET_STEP = 64
_T_BUCKET_MAX = 512  # 30 s * (1000 ms / 60 ms per LFR frame) ~= 500 frames


def _sdpa_capture_context() -> AbstractContextManager[None]:
    """Pin the SDPA backends the platform says a capture may use.

    An empty answer means this platform's default selection already captures,
    and pinning there would only cost the capture kernels it could pick from.
    """
    backends = current_platform.get_graph_capture_sdpa_backends()
    if not backends:
        return nullcontext()
    return sdpa_kernel(list(backends), set_priority=True)


def _graph_api(device: torch.device) -> tuple[Any, dict[str, Any]]:
    """Graph type and capture keywords for a device; (None, {}) if it cannot capture.

    Named per device rather than probed off the device module, so a device that
    grows graph support opts in here deliberately instead of being captured on
    the strength of an attribute name.
    """
    if device.type == "cuda":  # rocm reports cuda as well
        # note (wilsonzheng0327): thread_local error mode -- the LM scheduler
        # thread keeps launching kernels concurrently and must not poison this
        # thread's capture.
        return torch.cuda.CUDAGraph, {"capture_error_mode": "thread_local"}
    if device.type == "xpu":
        # torch.xpu.graph accepts no error mode, so nothing to pass.
        return torch.xpu.XPUGraph, {}
    return None, {}


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
        self._device_module = torch.get_device_module(self._device)
        self._graph_cls, self._capture_kwargs = _graph_api(self._device)
        if self._graph_cls is None:
            logger.info(
                "Fun-ASR encoder graphs are unavailable on %s; the encoder runs "
                "eager",
                self._device,
            )
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
        self._done_event = self._device_module.Event()
        self._event_recorded = False

    def _forward(self, xs: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        enc_out = self._audio_tower(xs, mask)
        return self._projector(enc_out, mask)

    def _enough_free_vram(self) -> tuple[bool, int]:
        free, total = self._device_module.mem_get_info(self._device)
        # Some accelerator runtimes report the whole card as free no matter what
        # is resident, so what this process already reserved is the ceiling on
        # what is left. On CUDA mem_get_info is the tighter of the two and wins.
        headroom = int(total) - int(self._device_module.memory_reserved(self._device))
        free = min(int(free), max(headroom, 0))
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

        # The warmup runs under the same attention backends the capture will
        # record, so what is captured is what was exercised. Naming backends is
        # process-global while held, hence scoped to this one capture.
        with _sdpa_capture_context():
            # note (wilsonzheng0327): warmup on a fresh stream so allocator state
            # settles before capture.
            stream = self._device_module.Stream(device=self._device)
            stream.wait_stream(self._device_module.current_stream())
            with self._device_module.stream(stream):
                for _ in range(self._warmup_iters):
                    _masked_forward()
            self._device_module.current_stream().wait_stream(stream)
            self._device_module.synchronize()

            if self._pool is None:
                self._pool = self._device_module.graph_pool_handle()
            graph = self._graph_cls()
            with self._device_module.graph(
                graph, pool=self._pool, **self._capture_kwargs
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
        if self._graph_cls is None:
            return None
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
                    with self._device_module.device(self._device):
                        entry = self._capture(batch_bucket, t_bucket, feat_dim)
                except Exception as exc:
                    # The traceback names the op that refused, which the message
                    # alone does not; a bucket only fails once before it is
                    # blacklisted, so this cannot repeat per request.
                    logger.warning(
                        "Fun-ASR encoder CUDA graph capture failed for "
                        "batch=%d t=%d: %s; using eager for this bucket",
                        batch_bucket,
                        t_bucket,
                        exc,
                        exc_info=True,
                    )
                    self._failed.add(key)
                    return None
                self._graphs[key] = entry

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
