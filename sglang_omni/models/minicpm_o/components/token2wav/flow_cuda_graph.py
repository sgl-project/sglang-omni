# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications: retain MiniCPM-o inference only; local imports and typing.
"""Capture and replay MiniCPM-o non-packed Flow CUDA graphs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
FLOW_CUDA_GRAPH_FRAME_BUCKET = 16


@dataclass(kw_only=True)
class CapturedFlowGraph:
    graph: torch.cuda.CUDAGraph
    inputs: tuple[torch.Tensor, ...]
    output: torch.Tensor


class FlowCudaGraphRunner:
    """Replay startup-captured non-packed Flow trajectories."""

    def __init__(
        self,
        decoder: torch.nn.Module,
        *,
        device: torch.device,
        min_free_gb: float = 3.0,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("Flow CUDA graphs require a CUDA device")
        else:
            pass
        self.decoder: torch.nn.Module = decoder
        self.device: torch.device = torch.device(
            "cuda",
            device.index if device.index is not None else torch.cuda.current_device(),
        )
        self.dtype: torch.dtype = next(decoder.parameters()).dtype
        self.min_free_bytes: int = int(min_free_gb * 1024**3)
        self.graphs: dict[tuple[int, int], CapturedFlowGraph] = {}
        self.pool: tuple[int, int] | None = None
        self.lock: Lock = Lock()
        self.graph_replays: int = 0
        self.graph_misses: int = 0

    @torch.inference_mode()
    def capture(self, shapes: tuple[tuple[int, int], ...]) -> None:
        if not shapes or len(set(shapes)) != len(shapes):
            raise ValueError("Flow CUDA graph shapes must be nonempty and unique")
        else:
            pass
        if any(
            len(shape) != 2
            or shape[0] <= 0
            or shape[1] <= 0
            or shape[1] % FLOW_CUDA_GRAPH_FRAME_BUCKET != 0
            or shape[1] > self.decoder.rand_noise.shape[2]
            for shape in shapes
        ):
            raise ValueError(
                "Flow CUDA graph shapes need positive batches and 16-aligned mel frames"
            )
        else:
            pass

        current_stream = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current_stream)
        graphs: dict[tuple[int, int], CapturedFlowGraph] = {}
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            self.pool = torch.cuda.graph_pool_handle()
            for batch_size, frames in sorted(
                shapes, key=lambda value: value[0] * value[1], reverse=True
            ):
                if batch_size > 1 and self.decoder.estimator.enable_variable_length:
                    continue
                else:
                    pass
                shape = (batch_size, frames)
                free, _ = torch.cuda.mem_get_info(self.device)
                if free < self.min_free_bytes:
                    logger.warning(
                        f"MiniCPM-o Flow CUDA graph skipped batch={batch_size} "
                        f"frames={frames}: free VRAM {free / 1024**3:.1f} GB "
                        f"is below {self.min_free_bytes / 1024**3:.1f} GB headroom"
                    )
                    continue
                else:
                    pass
                try:
                    x = (
                        self.decoder.rand_noise[:, :, :frames]
                        .expand(batch_size, -1, -1)
                        .clone()
                    )
                    dt = torch.tensor(0.1, device=self.device, dtype=self.dtype)
                    t = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
                    mu_in = torch.zeros(
                        2 * batch_size,
                        self.decoder.out_channels,
                        frames,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    mask_in = torch.ones(
                        2 * batch_size,
                        1,
                        frames,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    spks_in = torch.zeros(
                        2 * batch_size,
                        self.decoder.out_channels,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    cond_in = torch.zeros_like(mu_in)
                    inputs = (x, t, dt, mu_in, mask_in, spks_in, cond_in)
                    with torch.autocast(
                        "cuda",
                        dtype=self.dtype,
                        enabled=self.dtype != torch.float32,
                    ):
                        for _ in range(3):
                            self.decoder.euler_step(*inputs)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(
                            cuda_graph=graph,
                            pool=self.pool,
                            stream=stream,
                            capture_error_mode="thread_local",
                        ):
                            x.copy_(self.decoder.euler_step(*inputs))
                    graphs[shape] = CapturedFlowGraph(
                        graph=graph, inputs=inputs, output=x
                    )
                except Exception as exc:
                    logger.warning(
                        f"MiniCPM-o Flow CUDA graph capture failed for "
                        f"batch={batch_size} frames={frames}: {exc}; using eager"
                    )
        current_stream.wait_stream(stream)
        self.graphs = graphs
        logger.info(f"Captured {len(graphs)} MiniCPM-o Flow step CUDA graph shapes")

    def fit(self, x: torch.Tensor) -> tuple[int, int] | None:
        actual_frames = x.shape[2]
        bucket_frames = (
            (actual_frames + FLOW_CUDA_GRAPH_FRAME_BUCKET - 1)
            // FLOW_CUDA_GRAPH_FRAME_BUCKET
            * FLOW_CUDA_GRAPH_FRAME_BUCKET
        )
        key = min(
            (
                key
                for key in self.graphs
                if key[0] == x.shape[0] and key[1] >= bucket_frames
            ),
            key=lambda value: value[1],
            default=None,
        )
        if x.device != self.device or x.dtype != self.dtype or key is None:
            self.graph_misses += 1
            return None
        else:
            pass
        return key

    def prepare(
        self,
        key: tuple[int, int],
        x: torch.Tensor,
        mu_in: torch.Tensor,
        mask_in: torch.Tensor,
        spks_in: torch.Tensor,
        cond_in: torch.Tensor,
    ) -> CapturedFlowGraph:
        """Initialize state and conditioning once while the solver holds the lock."""
        actual_frames = x.shape[2]
        captured = self.graphs[key]
        for target, source in zip(
            (captured.inputs[0], *captured.inputs[3:]),
            (x, mu_in, mask_in, spks_in, cond_in),
            strict=True,
        ):
            value = (
                F.pad(source, (0, key[1] - actual_frames))
                if source.ndim == 3 and source.shape[-1] != key[1]
                else source
            )
            target.copy_(value)
        return captured

    def run_step(
        self,
        captured: CapturedFlowGraph,
        t: torch.Tensor,
        dt: torch.Tensor,
    ) -> None:
        """Update time inputs and advance graph-owned state."""
        captured.inputs[1].copy_(t)
        captured.inputs[2].copy_(dt)
        try:
            captured.graph.replay()
        except RuntimeError:
            self.graphs.clear()
            self.pool = None
            logger.exception("MiniCPM-o Flow CUDA graph replay disabled the runner")
            raise
        self.graph_replays += 1
