# SPDX-License-Identifier: Apache-2.0
"""Device graph replay for the fixed-shape MiniMax RVQ depth pass."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import DeviceGraphBackend

logger = logging.getLogger(__name__)

RVQDepthForward = Callable[..., tuple[Tensor, ...]]
_WARMUP_REPLAYS = 3


class RVQDepthCudaGraphRunner:
    """Capture the seven depth-code steps, sampling included, as one graph."""

    def __init__(
        self,
        *,
        forward: RVQDepthForward,
        backend: DeviceGraphBackend,
        device: torch.device,
        dtype: torch.dtype,
        hidden_size: int,
        num_codebooks: int,
        buckets: list[int],
    ) -> None:
        self.forward = forward
        self.backend = backend
        self.device = device
        self.device_module = torch.get_device_module(device)
        self.buckets = sorted(set(buckets))
        if not self.buckets or self.buckets[0] < 1:
            raise ValueError("MiniMax Music 3 RVQ graph needs positive batch buckets")
        else:
            pass
        self.graphs: dict[int, Any] = {}
        self.inputs: dict[int, tuple[Tensor, ...]] = {}
        self.outputs: dict[int, tuple[Tensor, Tensor, Tensor]] = {}
        self.allocated_bytes = 0
        for size in self.buckets:
            self.capture(size, dtype, hidden_size, num_codebooks)
        logger.info(
            f"MiniMax Music 3 RVQ depth device graphs captured buckets={self.buckets} memory={self.allocated_bytes / 2 ** 20:.1f}MiB"
        )

    @property
    def max_batch_size(self) -> int:
        return self.buckets[-1]

    @torch.inference_mode()
    def capture(
        self, size: int, dtype: torch.dtype, hidden_size: int, num_codebooks: int
    ) -> None:
        hidden = torch.zeros((size, hidden_size), device=self.device, dtype=dtype)
        c0 = torch.zeros((size,), device=self.device, dtype=torch.long)
        seeds = torch.ones((size,), device=self.device, dtype=torch.int64)
        positions = torch.zeros((size,), device=self.device, dtype=torch.int64)
        forced = torch.zeros(
            (size, num_codebooks), device=self.device, dtype=torch.long
        )
        replay = torch.zeros((), device=self.device, dtype=torch.bool)

        # The depth decoder's attention must reach a capturable kernel: XPU's
        # default SDPA path enqueues work on events the graph cannot own.
        with (
            self.device_module.device(self.device),
            current_platform.graph_capture_attention(),
        ):
            for _ in range(_WARMUP_REPLAYS):
                self.forward(hidden, c0, seeds, positions, forced, replay)
            self.device_module.synchronize(self.device)
            allocated_before = self.device_module.memory_allocated(self.device)
            with self.backend.capture() as graph:
                outputs = self.forward(hidden, c0, seeds, positions, forced, replay)
            self.device_module.synchronize(self.device)
            self.allocated_bytes += max(
                0, self.device_module.memory_allocated(self.device) - allocated_before
            )

        self.graphs[size] = graph
        self.inputs[size] = (hidden, c0, seeds, positions, forced, replay)
        self.outputs[size] = outputs

    def bucket_for(self, rows: int) -> int | None:
        for size in self.buckets:
            if size >= rows:
                return size
            else:
                pass
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
        size = self.bucket_for(rows)
        if size is None:
            return None
        else:
            pass
        statics = self.inputs[size]
        static_hidden, static_c0, static_seeds, static_positions = statics[:4]
        static_forced, static_replay = statics[4:]
        static_hidden[:rows].copy_(hidden)
        static_c0[:rows].copy_(c0)
        static_seeds[:rows].copy_(seeds)
        static_positions[:rows].copy_(positions)
        static_forced[:rows].copy_(forced)
        static_replay.copy_(replay)
        self.graphs[size].replay()
        return tuple(output[:rows].clone() for output in self.outputs[size])


__all__ = ["RVQDepthCudaGraphRunner"]
