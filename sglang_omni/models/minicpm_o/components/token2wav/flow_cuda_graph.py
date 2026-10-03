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
"""Capture and replay MiniCPM-o dense Euler steps after DiT compilation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

import torch
import torch.nn.functional as F
from torch._dynamo.exc import TorchDynamoException

logger = logging.getLogger(__name__)


class FlowEulerStep(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        paired_mu: torch.Tensor,
        paired_mask: torch.Tensor,
        paired_speaker_embeddings: torch.Tensor,
        paired_mel_conditioning: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass(kw_only=True)
class CapturedFlowGraph:
    graph: torch.cuda.CUDAGraph
    inputs: tuple[torch.Tensor, ...]
    output: torch.Tensor


class FlowCudaGraphRunner:
    def __init__(
        self,
        euler_step: FlowEulerStep,
        initial_noise: torch.Tensor,
        *,
        minimum_free_gibibytes: float = 3.0,
    ) -> None:
        self.euler_step: FlowEulerStep = euler_step
        self.initial_noise: torch.Tensor = initial_noise
        self.device: torch.device = initial_noise.device
        self.dtype: torch.dtype = initial_noise.dtype
        self.minimum_free_bytes: int = int(minimum_free_gibibytes * 1024**3)
        self.graphs: dict[tuple[int, int], CapturedFlowGraph] = {}
        self.pool: tuple[int, int] | None = None
        self.lock: Lock = Lock()
        self.graph_replays: int = 0
        self.graph_misses: int = 0

    def capture_inputs(self, batch_size: int, frames: int) -> tuple[torch.Tensor, ...]:
        x = self.initial_noise[:, :, :frames].expand(batch_size, -1, -1).clone()
        # note (wirybeaver): The conformer emits FP32 conditioning under FP16 autocast.
        t = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        dt = torch.tensor(0.1, device=self.device, dtype=torch.float32)
        mu = torch.zeros(
            2 * batch_size,
            self.initial_noise.shape[1],
            frames,
            device=self.device,
            dtype=torch.float32,
        )
        mask = torch.ones(2 * batch_size, 1, frames, device=self.device)
        speaker_embeddings = torch.zeros(
            2 * batch_size, self.initial_noise.shape[1], device=self.device
        )
        mel_conditioning = torch.zeros_like(mu)
        return x, t, dt, mu, mask, speaker_embeddings, mel_conditioning

    @torch.inference_mode()
    def capture(self, shapes: tuple[tuple[int, int], ...]) -> None:
        if len(set(shapes)) != len(shapes) or any(
            batch_size <= 0 or not 0 < frames <= self.initial_noise.shape[2]
            for batch_size, frames in shapes
        ):
            raise ValueError(
                "Flow graph shapes require unique positive batches and supported mel lengths"
            )
        else:
            pass
        current_stream = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current_stream)
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            self.pool = torch.cuda.graph_pool_handle()
            for batch_size, frames in sorted(
                shapes, key=lambda shape: shape[0] * shape[1], reverse=True
            ):
                free_bytes, _ = torch.cuda.mem_get_info(self.device)
                if free_bytes < self.minimum_free_bytes:
                    logger.warning(
                        f"Skipping Flow graph {(batch_size, frames)}: insufficient VRAM headroom"
                    )
                    continue
                else:
                    pass
                try:
                    inputs = self.capture_inputs(batch_size, frames)
                    with torch.autocast(
                        "cuda", dtype=self.dtype, enabled=self.dtype != torch.float32
                    ):
                        for _ in range(3):
                            self.euler_step(*inputs)
                except torch.cuda.OutOfMemoryError as exc:
                    logger.warning(
                        f"Flow graph warmup failed for {(batch_size, frames)}: {exc}; using eager"
                    )
                    continue
                try:
                    graph = torch.cuda.CUDAGraph()
                    with (
                        torch.cuda.graph(
                            graph,
                            pool=self.pool,
                            stream=stream,
                            capture_error_mode="thread_local",
                        ),
                        torch.autocast(
                            "cuda",
                            dtype=self.dtype,
                            enabled=self.dtype != torch.float32,
                        ),
                    ):
                        inputs[0].copy_(self.euler_step(*inputs))
                except TorchDynamoException:
                    raise
                except RuntimeError as exc:
                    logger.warning(
                        f"Flow graph capture failed for {(batch_size, frames)}: {exc}; using eager"
                    )
                    continue
                self.graphs[(batch_size, frames)] = CapturedFlowGraph(
                    graph=graph, inputs=inputs, output=inputs[0]
                )
        current_stream.wait_stream(stream)
        logger.info(f"Captured {len(self.graphs)} MiniCPM-o dense Flow graphs")

    def fit(self, x: torch.Tensor, time_span: torch.Tensor) -> tuple[int, int] | None:
        batch_frame_shape = min(
            (
                shape
                for shape in self.graphs
                if shape[0] == x.shape[0] and shape[1] >= x.shape[2]
            ),
            key=lambda shape: shape[1],
            default=None,
        )
        if (
            x.device != self.device
            or x.dtype != self.dtype
            or time_span.dtype != torch.float32
            or batch_frame_shape is None
        ):
            self.graph_misses += 1
            return None
        else:
            return batch_frame_shape

    def prepare(
        self,
        batch_frame_shape: tuple[int, int],
        x: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        mel_conditioning: torch.Tensor,
    ) -> CapturedFlowGraph:
        captured = self.graphs[batch_frame_shape]
        for target, source in zip(
            (captured.inputs[0], *captured.inputs[3:]),
            (x, mu, mask, speaker_embeddings, mel_conditioning),
            strict=True,
        ):
            target.copy_(
                F.pad(source, (0, batch_frame_shape[1] - x.shape[2]))
                if source.ndim == 3
                else source
            )
        return captured

    def run_step(
        self, captured: CapturedFlowGraph, t: torch.Tensor, dt: torch.Tensor
    ) -> None:
        captured.inputs[1].copy_(t)
        captured.inputs[2].copy_(dt)
        captured.graph.replay()
        self.graph_replays += 1
