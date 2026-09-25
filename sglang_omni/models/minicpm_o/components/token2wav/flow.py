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
"""Flow for MiniCPM-o."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from threading import Lock
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from sglang_omni.models.minicpm_o.components.token2wav.conformer import (
    UpsampleConformerEncoderV2,
    make_pad_mask,
)
from sglang_omni.models.minicpm_o.components.token2wav.dit import (
    CapturedPackedDiTGraph,
    DiT,
)
from sglang_omni.models.minicpm_o.components.token2wav.fixed_packed import (
    FixedPackedLayout,
    build_fixed_packed_layout,
)

logger = logging.getLogger(__name__)
FLOW_CUDA_GRAPH_FRAME_BUCKET = 16


class CausalConditionalCFM(torch.nn.Module):

    def __init__(self, estimator: DiT, inference_cfg_rate: float = 0.7) -> None:
        super().__init__()
        self.estimator = estimator
        self.inference_cfg_rate = inference_cfg_rate
        self.out_channels = estimator.out_channels
        self.graph_runner: FlowCudaGraphRunner | None = None
        self.register_buffer(
            "rand_noise",
            torch.randn([1, self.out_channels, 50 * 600]),
            persistent=False,
        )

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        *,
        packed_valid_frames: int | None = None,
    ) -> torch.Tensor:
        t = t_span[0].expand(x.size(0))
        dt = t_span[1] - t_span[0]
        assert self.inference_cfg_rate > 0, "inference_cfg_rate better > 0"
        mask_in = torch.cat([mask, mask], dim=0)
        mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
        cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        graph_runner = self.graph_runner
        packed_runner = self.estimator.packed_graph_runner
        is_packed = self.estimator.enable_variable_length and x.shape[0] > 1
        graph_key = (
            graph_runner.fit(x) if graph_runner is not None and not is_packed else None
        )
        packed_capacity = (
            packed_runner.fit(x.shape[0], x.shape[2], packed_valid_frames)
            if packed_runner is not None and is_packed
            else None
        )
        packed_layout = (
            build_fixed_packed_layout(
                mask_in.bool().squeeze(1).sum(dim=1, dtype=torch.int32),
                x.shape[2],
                packed_capacity,
                self.estimator.blocks[0].conv.kernel_size - 1,
            )
            if packed_capacity is not None
            else None
        )
        if graph_key is not None:
            active_runner = graph_runner
        elif packed_layout is not None:
            active_runner = packed_runner
        else:
            active_runner = None
        with active_runner.lock if active_runner is not None else nullcontext():
            captured_flow = (
                graph_runner.prepare(graph_key, x, mu_in, mask_in, spks_in, cond_in)
                if graph_key is not None
                else None
            )
            captured_packed = (
                packed_runner.prepare(packed_layout)
                if packed_layout is not None
                else None
            )
            for step in range(1, len(t_span)):
                if captured_flow is not None:
                    graph_runner.run_step(captured_flow, t, dt)
                else:
                    x = self.euler_step(
                        x,
                        t,
                        dt,
                        mu_in,
                        mask_in,
                        spks_in,
                        cond_in,
                        packed_layout=packed_layout,
                        packed_graph=captured_packed,
                    )
                t = t + dt
                if step < len(t_span) - 1:
                    dt = t_span[step + 1] - t_span[step]
                else:
                    pass
            return (
                captured_flow.output[:, :, : x.shape[2]].clone()
                if captured_flow is not None
                else x
            )

    def euler_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        mu_in: torch.Tensor,
        mask_in: torch.Tensor,
        spks_in: torch.Tensor,
        cond_in: torch.Tensor,
        *,
        packed_layout: FixedPackedLayout | None = None,
        packed_graph: CapturedPackedDiTGraph | None = None,
    ) -> torch.Tensor:
        dphi_dt = self.estimator.forward(
            torch.cat([x, x], dim=0),
            mask_in,
            mu_in,
            torch.cat([t, t], dim=0),
            spks_in,
            cond_in,
            packed_layout=packed_layout,
            packed_graph=packed_graph,
        )
        conditional, unconditional = dphi_dt.chunk(2, dim=0)
        velocity = (
            1.0 + self.inference_cfg_rate
        ) * conditional - self.inference_cfg_rate * unconditional
        return x + dt * velocity

    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        n_timesteps: int = 10,
        temperature: float = 1.0,
        packed_valid_frames: int | None = None,
    ) -> torch.Tensor:
        if n_timesteps <= 0:
            raise ValueError("n_timesteps must be positive")
        else:
            pass
        if mu.size(2) > self.rand_noise.size(2):
            raise ValueError(
                "Combined reference and generated audio exceed 600 seconds"
            )
        else:
            pass
        z = (
            self.rand_noise[:, :, : mu.size(2)].expand(mu.size(0), -1, -1).clone()
            * temperature
        )
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(
            z,
            t_span,
            mu,
            mask,
            spks,
            cond,
            packed_valid_frames=packed_valid_frames,
        )


@dataclass(kw_only=True)
class CapturedFlowGraph:
    graph: torch.cuda.CUDAGraph
    inputs: tuple[torch.Tensor, ...]
    output: torch.Tensor


class FlowCudaGraphRunner:
    """Replay startup-captured non-packed Flow trajectories."""

    def __init__(
        self,
        decoder: CausalConditionalCFM,
        *,
        device: torch.device,
        min_free_gb: float = 3.0,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("Flow CUDA graphs require a CUDA device")
        else:
            pass
        self.decoder: CausalConditionalCFM = decoder
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


class CausalMaskedDiffWithXvec(torch.nn.Module):

    def __init__(
        self,
        encoder: UpsampleConformerEncoderV2,
        decoder: CausalConditionalCFM,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: Literal["mel"] = "mel",
        vocab_size: int = 6561,
    ) -> None:
        super().__init__()
        if output_type != "mel":
            raise ValueError("MiniCPM-o flow output must be mel")
        else:
            pass
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.pre_lookahead_len = int(encoder.pre_lookahead_layer.pre_lookahead_len)
        self.up_rate = int(encoder.up_layer.stride)
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_dim, output_size)
        self.decoder = decoder

    @torch.inference_mode()
    def inference(
        self,
        token: torch.Tensor,
        token_len: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_token_len: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        n_timesteps: int = 10,
    ) -> torch.Tensor:
        assert token.shape[0] == prompt_token.shape[0], (
            f"flow batch size mismatch: token={token.shape[0]} "
            f"prompt_token={prompt_token.shape[0]}"
        )
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        batch_size = token.shape[0]
        prompt_token_lens = [int(prompt_token_len[i]) for i in range(batch_size)]
        token_lens = [int(token_len[i]) for i in range(batch_size)]
        # Rows may carry references of different lengths, so build each row as
        # prompt-then-generated and pad the tail. The mask drops that tail, and
        # each row's prompt mel is written at its own offset below.
        combined = pad_sequence(
            [
                torch.cat(
                    [
                        prompt_token[i, : prompt_token_lens[i]],
                        token[i, : token_lens[i]],
                    ]
                )
                for i in range(batch_size)
            ],
            batch_first=True,
        )
        token_len = prompt_token_len + token_len
        token_mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(combined, min=0)) * token_mask
        h, _ = self.encoder.forward(token, token_len)
        frame_mask = (~make_pad_mask(token_len * self.up_rate, h.shape[1])).to(h)
        h = self.encoder_proj(h) * frame_mask.unsqueeze(-1)
        conds = torch.zeros_like(h)
        for i, prompt_len in enumerate(prompt_token_lens):
            prompt_frames = prompt_len * self.up_rate
            conds[i, :prompt_frames] = prompt_feat[i, :prompt_frames]
        conds = conds.transpose(1, 2).contiguous()
        # note (wirybeaver): CFG duplicates every valid row before DiT packing.
        packed_valid_frames = (
            2
            * sum(
                min((prompt_len + generated_len) * self.up_rate, h.shape[1])
                for prompt_len, generated_len in zip(
                    prompt_token_lens, token_lens, strict=True
                )
            )
            if batch_size > 1
            else None
        )
        feat = self.decoder.forward(
            mu=h.transpose(1, 2).contiguous(),
            mask=frame_mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
            packed_valid_frames=packed_valid_frames,
        )
        generated = [
            feat[i, :, prompt_token_lens[i] * self.up_rate :][
                :, : token_lens[i] * self.up_rate
            ]
            for i in range(batch_size)
        ]
        return pad_sequence(
            [row.transpose(0, 1) for row in generated], batch_first=True
        ).transpose(1, 2)
