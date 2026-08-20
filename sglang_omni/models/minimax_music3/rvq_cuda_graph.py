# SPDX-License-Identifier: Apache-2.0
"""CUDA Graph replay for the fixed-shape MiniMax RVQ depth pass."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

RVQDepthForward = Callable[..., tuple[Tensor, ...]]
_WARMUP_REPLAYS = 3


class RVQDepthCudaGraphRunner:
    """Capture the seven depth-code steps, sampling included, as one graph."""

    def __init__(
        self,
        *,
        forward: RVQDepthForward,
        device: torch.device,
        dtype: torch.dtype,
        hidden_size: int,
        num_codebooks: int,
        buckets: list[int],
    ) -> None:
        self._forward = forward
        self._device = device
        self._buckets = sorted(set(buckets))
        if not self._buckets or self._buckets[0] < 1:
            raise ValueError("MiniMax Music 3 RVQ graph needs positive batch buckets")
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._inputs: dict[int, tuple[Tensor, ...]] = {}
        self._outputs: dict[int, tuple[Tensor, Tensor, Tensor]] = {}
        self.allocated_bytes = 0
        for size in self._buckets:
            self._capture(size, dtype, hidden_size, num_codebooks)
        logger.info(
            f"MiniMax Music 3 RVQ depth CUDA graphs captured buckets={self._buckets} memory={self.allocated_bytes / 2 ** 20:.1f}MiB"
        )

    @property
    def max_batch_size(self) -> int:
        return self._buckets[-1]

    @torch.inference_mode()
    def _capture(
        self, size: int, dtype: torch.dtype, hidden_size: int, num_codebooks: int
    ) -> None:
        hidden = torch.zeros((size, hidden_size), device=self._device, dtype=dtype)
        c0 = torch.zeros((size,), device=self._device, dtype=torch.long)
        seeds = torch.ones((size,), device=self._device, dtype=torch.int64)
        positions = torch.zeros((size,), device=self._device, dtype=torch.int64)
        forced = torch.zeros(
            (size, num_codebooks), device=self._device, dtype=torch.long
        )
        replay = torch.zeros((), device=self._device, dtype=torch.bool)

        with torch.cuda.device(self._device):
            for _ in range(_WARMUP_REPLAYS):
                self._forward(hidden, c0, seeds, positions, forced, replay)
            torch.cuda.synchronize(self._device)
            allocated_before = torch.cuda.memory_allocated(self._device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = self._forward(hidden, c0, seeds, positions, forced, replay)
            torch.cuda.synchronize(self._device)
            self.allocated_bytes += max(
                0, torch.cuda.memory_allocated(self._device) - allocated_before
            )

        self._graphs[size] = graph
        self._inputs[size] = (hidden, c0, seeds, positions, forced, replay)
        self._outputs[size] = outputs

    def _bucket_for(self, rows: int) -> int | None:
        for size in self._buckets:
            if size >= rows:
                return size
        return None

    @torch.inference_mode()
    def __call__(
        self,
        hidden: Tensor,
        c0: Tensor,
        seeds: Tensor,
        positions: Tensor,
        forced: Tensor,
        replay: Tensor,
    ) -> tuple[Tensor, ...] | None:
        """Replay for this batch, or None when no captured bucket fits."""
        rows = hidden.shape[0]
        size = self._bucket_for(rows)
        if size is None:
            return None
        statics = self._inputs[size]
        static_hidden, static_c0, static_seeds, static_positions = statics[:4]
        static_forced, static_replay = statics[4:]
        static_hidden[:rows].copy_(hidden)
        static_c0[:rows].copy_(c0)
        static_seeds[:rows].copy_(seeds)
        static_positions[:rows].copy_(positions)
        static_forced[:rows].copy_(forced)
        static_replay.copy_(replay)
        self._graphs[size].replay()
        return tuple(output[:rows].clone() for output in self._outputs[size])


__all__ = ["RVQDepthCudaGraphRunner"]
