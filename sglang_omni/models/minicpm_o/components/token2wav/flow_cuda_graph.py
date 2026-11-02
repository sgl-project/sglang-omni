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
"""Capture dense Flow steps and the dense boundaries of packed Flow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

import torch
import torch.nn.functional as F
from torch._dynamo.exc import TorchDynamoException

from sglang_omni.models.minicpm_o.components.token2wav.dit import DiT
from sglang_omni.models.minicpm_o.components.token2wav.fixed_packed import (
    FixedPackedLayout,
)
from sglang_omni.models.minicpm_o.components.token2wav.packed_dit_cuda_graph import (
    CapturedPackedDiTGraph,
    PackedDiTCudaGraphRunner,
)

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


@dataclass(kw_only=True)
class CapturedPackedFlowGraph:
    pre_graph: torch.cuda.CUDAGraph
    post_graph: torch.cuda.CUDAGraph
    inputs: tuple[torch.Tensor, ...]
    output: torch.Tensor
    positions: torch.Tensor
    padding_mask: torch.Tensor


class FlowCudaGraphRunner:
    """Replay dense Euler steps or dense boundaries around packed DiT."""

    def __init__(
        self,
        euler_step: FlowEulerStep,
        initial_noise: torch.Tensor,
        *,
        estimator: DiT,
        inference_cfg_rate: float,
        minimum_free_gibibytes: float = 3.0,
    ) -> None:
        self.estimator: DiT = estimator
        self.inference_cfg_rate: float = inference_cfg_rate
        self.packed_step_replays: int = 0
        self.euler_step: FlowEulerStep = euler_step
        self.initial_noise: torch.Tensor = initial_noise
        self.device: torch.device = initial_noise.device
        self.dtype: torch.dtype = initial_noise.dtype
        self.minimum_free_bytes: int = int(minimum_free_gibibytes * 1024**3)
        self.graphs: dict[
            tuple[int, int], CapturedFlowGraph | CapturedPackedFlowGraph
        ] = {}
        self.pool: tuple[int, int] | None = None
        self.lock: Lock = Lock()
        self.graph_replays: int = 0
        self.graph_misses: int = 0

    def capture_inputs(
        self, batch_size: int, frames: int, *, is_packed: bool = False
    ) -> tuple[torch.Tensor, ...]:
        x = self.initial_noise[:, :, :frames].expand(batch_size, -1, -1).clone()
        if is_packed:
            x = x.to(torch.promote_types(self.dtype, torch.bfloat16))
        else:
            pass
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
        graphs: dict[tuple[int, int], CapturedFlowGraph | CapturedPackedFlowGraph] = {}
        packed_runner = self.estimator.packed_graph_runner
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            self.pool = torch.cuda.graph_pool_handle()
            for batch_size, frames in sorted(
                shapes, key=lambda value: value[0] * value[1], reverse=True
            ):
                is_packed = batch_size > 1 and self.estimator.enable_variable_length
                if is_packed and (
                    packed_runner is None
                    or not any(batch == batch_size for batch, _ in packed_runner.graphs)
                    or 2 * batch_size * frames
                    > packed_runner.workspaces[batch_size].dense_input.shape[0]
                ):
                    continue
                else:
                    pass
                shape = (batch_size, frames)
                free_bytes, _ = torch.cuda.mem_get_info(self.device)
                if free_bytes < self.minimum_free_bytes:
                    logger.warning(
                        f"MiniCPM-o Flow CUDA graph skipped batch={batch_size} "
                        f"frames={frames}: free VRAM {free_bytes / 1024**3:.1f} GB "
                        f"is below {self.minimum_free_bytes / 1024**3:.1f} GB headroom"
                    )
                    continue
                else:
                    pass
                try:
                    inputs = self.capture_inputs(
                        batch_size, frames, is_packed=is_packed
                    )
                    if is_packed:
                        positions = torch.zeros(
                            2 * batch_size * frames,
                            device=self.device,
                            dtype=torch.long,
                        )
                        padding_mask = torch.zeros_like(positions, dtype=torch.bool)
                    else:
                        pass
                    (
                        x,
                        t,
                        dt,
                        paired_mu,
                        paired_mask,
                        paired_speaker_embeddings,
                        paired_mel_conditioning,
                    ) = inputs
                    if is_packed:
                        workspace = packed_runner.workspaces[batch_size]

                        def run_pre() -> None:
                            with torch.autocast(
                                "cuda",
                                dtype=self.dtype,
                                enabled=self.dtype != torch.float32,
                            ):
                                projected, conditioning = self.estimator.prepare_inputs(
                                    torch.cat([x, x]),
                                    paired_mu,
                                    torch.cat([t, t]),
                                    paired_speaker_embeddings,
                                    paired_mel_conditioning,
                                )
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                projected = self.estimator.in_proj(projected).to(
                                    torch.bfloat16
                                )
                            workspace.dense_input[: 2 * batch_size * frames].copy_(
                                projected.flatten(0, 1)
                            )
                            workspace.conditioning[:-1].copy_(conditioning.squeeze(1))

                        def run_post() -> None:
                            prediction = torch.index_select(
                                workspace.packed_output, 0, positions
                            )
                            prediction.masked_fill_(padding_mask[:, None], 0)
                            prediction = prediction.reshape(
                                2 * batch_size, frames, self.initial_noise.shape[1]
                            ).transpose(1, 2)
                            conditional, unconditional = prediction.chunk(2)
                            velocity = (
                                (1.0 + self.inference_cfg_rate) * conditional
                                - self.inference_cfg_rate * unconditional
                            )
                            x.copy_(x + dt * velocity)

                        for _ in range(3):
                            run_pre()
                            run_post()
                    else:
                        with torch.autocast(
                            "cuda",
                            dtype=self.dtype,
                            enabled=self.dtype != torch.float32,
                        ):
                            for _ in range(3):
                                self.euler_step(*inputs)
                except torch.cuda.OutOfMemoryError as exc:
                    logger.warning(
                        f"MiniCPM-o Flow graph preparation failed for "
                        f"batch={batch_size} frames={frames}: {exc}; using eager"
                    )
                    continue

                try:
                    if is_packed:
                        pre_graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(
                            pre_graph,
                            pool=self.pool,
                            stream=stream,
                            capture_error_mode="thread_local",
                        ):
                            run_pre()
                        post_graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(
                            post_graph,
                            pool=self.pool,
                            stream=stream,
                            capture_error_mode="thread_local",
                        ):
                            run_post()
                        graphs[shape] = CapturedPackedFlowGraph(
                            pre_graph=pre_graph,
                            post_graph=post_graph,
                            inputs=inputs,
                            output=x,
                            positions=positions,
                            padding_mask=padding_mask,
                        )
                    else:
                        with torch.autocast(
                            "cuda",
                            dtype=self.dtype,
                            enabled=self.dtype != torch.float32,
                        ):
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(
                                cuda_graph=graph,
                                pool=self.pool,
                                stream=stream,
                                capture_error_mode="thread_local",
                            ):
                                x.copy_(self.euler_step(*inputs))
                        graphs[shape] = CapturedFlowGraph(
                            graph=graph, inputs=inputs, output=x
                        )
                except TorchDynamoException:
                    raise
                except RuntimeError as exc:
                    logger.warning(
                        f"MiniCPM-o Flow CUDA graph capture failed for "
                        f"batch={batch_size} frames={frames}: {exc}; using eager"
                    )
        current_stream.wait_stream(stream)
        self.graphs = graphs
        packed_graph_count = sum(
            isinstance(captured, CapturedPackedFlowGraph)
            for captured in graphs.values()
        )
        logger.info(
            f"Captured {len(graphs) - packed_graph_count} MiniCPM-o dense Flow graphs "
            f"and {packed_graph_count} packed Flow pre/post pairs"
        )

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
        packed_layout: FixedPackedLayout | None = None,
    ) -> CapturedFlowGraph | CapturedPackedFlowGraph:
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
        if isinstance(captured, CapturedPackedFlowGraph):
            assert packed_layout is not None
            captured.positions.copy_(packed_layout.positions)
            captured.padding_mask.copy_(packed_layout.padding_mask.flatten())
        else:
            pass
        return captured

    def run_packed_step(
        self,
        captured: CapturedPackedFlowGraph,
        packed_runner: PackedDiTCudaGraphRunner,
        packed_graph: CapturedPackedDiTGraph,
        t: torch.Tensor,
        dt: torch.Tensor,
    ) -> None:
        captured.inputs[1].copy_(t)
        captured.inputs[2].copy_(dt)
        captured.pre_graph.replay()
        packed_runner.replay(packed_graph)
        captured.post_graph.replay()
        self.graph_replays += 2
        self.packed_step_replays += 1

    def run_step(
        self, captured: CapturedFlowGraph, t: torch.Tensor, dt: torch.Tensor
    ) -> None:
        captured.inputs[1].copy_(t)
        captured.inputs[2].copy_(dt)
        captured.graph.replay()
        self.graph_replays += 1
