# SPDX-License-Identifier: Apache-2.0
"""One graph pool and capture stream for the Flow's Euler solve graphs."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Protocol

import torch
from sglang.srt.model_executor.runner.base_cuda_graph_runner import freeze_gc
from sglang.srt.utils.common import get_available_gpu_memory

from sglang_omni.platforms.device_graph import DeviceGraphBackend, ReplayableGraph

logger = logging.getLogger(__name__)

CAPTURE_WARMUP_RUNS = 2


class GraphSolve(Protocol):
    def __call__(self) -> torch.Tensor: ...


class GroupReplay(Protocol):
    def __call__(self, rows: slice, frames: slice) -> torch.Tensor: ...


def replay_in_groups(
    replay: GroupReplay, lengths: Sequence[int], *, max_frames: int, max_rows: int
) -> torch.Tensor:
    """replay over the fewest consecutive groups of rows whose frames sum to at most
    max_frames and whose count is at most max_rows, the outputs joined along the
    frames. Every length is at most max_frames."""
    generated: list[torch.Tensor] = []
    start = 0
    frame_start = 0
    frames = 0
    for index, length in enumerate(lengths):
        if frames + length > max_frames or index - start == max_rows:
            generated.append(
                replay(slice(start, index), slice(frame_start, frame_start + frames))
            )
            start = index
            frame_start += frames
            frames = 0
        else:
            pass
        frames += length
    generated.append(
        replay(slice(start, len(lengths)), slice(frame_start, frame_start + frames))
    )
    return torch.cat(generated, dim=1)


class SolveGraphCapture:
    """Captures the prefix hop and whole history solve graphs into one pool. They
    share it because a stream step replays one graph at a time, under the scheduler's
    state lock, and clones its output before the next replay."""

    def __init__(
        self,
        backend: DeviceGraphBackend,
        *,
        device: torch.device,
        autocast_dtype: torch.dtype,
    ) -> None:
        self.backend = backend
        self.device = device
        self.device_module = torch.get_device_module(device)
        self.autocast_dtype = autocast_dtype
        self.pool = backend.graph_pool_handle()
        self.stream = self.device_module.Stream(device=device)

    @contextmanager
    def capture_session(self, name: str, tier_frames: Sequence[int]) -> Iterator[None]:
        """One runner's captures with the collector frozen, their time and memory
        logged."""
        started = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device.type, self.device.index)
        # note (ratish): a collection during capture can free what a reference cycle
        # holds, an onnxruntime session among them, and invalidate the graph.
        with freeze_gc(enable_cudagraph_gc=False):
            yield
        after_mem = get_available_gpu_memory(self.device.type, self.device.index)
        logger.info(
            f"Fun-CosyVoice3 {name} solve graphs captured: tiers={list(tier_frames)} "
            f"frames, elapsed={time.perf_counter() - started:.2f} s, "
            f"mem usage={before_mem - after_mem:.2f} GB, avail mem={after_mem:.2f} GB."
        )

    def capture(self, solve: GraphSolve) -> tuple[ReplayableGraph, torch.Tensor]:
        """solve warmed on the capture stream, then captured into the shared pool.
        Returns the graph and the output its replays write."""
        current_stream = self.device_module.current_stream(self.device)
        self.stream.wait_stream(current_stream)
        with (
            self.device_module.stream(self.stream),
            torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype),
        ):
            for _ in range(CAPTURE_WARMUP_RUNS):
                solve()
        current_stream.wait_stream(self.stream)
        self.device_module.synchronize(self.device)
        with (
            self.backend.capture(
                pool=self.pool, stream=self.stream, thread_local_errors=True
            ) as graph,
            torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype),
        ):
            output = solve()
        self.device_module.synchronize(self.device)
        return graph, output
