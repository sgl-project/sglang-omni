# SPDX-License-Identifier: Apache-2.0
"""Startup CUDA Graph capture of the complete MiniCPM-o Euler solver."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import torch

logger = logging.getLogger(__name__)
WARMUP_ITERATIONS = 3


class EulerSolver(Protocol):
    rand_noise: torch.Tensor
    out_channels: int

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class GraphKey:
    batch_size: int
    mel_frames_bucket: int


@dataclass(kw_only=True)
class CapturedFlowGraph:
    graph: torch.cuda.CUDAGraph
    static_noise: torch.Tensor
    static_t_span: torch.Tensor
    static_mu: torch.Tensor
    static_mask: torch.Tensor
    static_spks: torch.Tensor
    static_cond: torch.Tensor
    static_output: torch.Tensor


class FlowCudaGraphRunner:
    def __init__(
        self,
        decoder: EulerSolver,
        *,
        capture_shapes: tuple[tuple[int, int], ...],
        frame_bucket: int,
        n_timesteps: int,
        conditioning_dtype: torch.dtype,
    ) -> None:
        if frame_bucket <= 0 or n_timesteps <= 0:
            raise ValueError("frame_bucket and n_timesteps must be positive")
        elif not capture_shapes or any(
            batch_size <= 0
            or frames <= 0
            or frames % frame_bucket
            or frames > decoder.rand_noise.shape[-1]
            for batch_size, frames in capture_shapes
        ):
            raise ValueError(
                "Capture shapes must have positive batches and bucket-aligned frames within the noise limit"
            )
        else:
            self.keys = tuple(
                sorted(
                    {GraphKey(*shape) for shape in capture_shapes},
                    key=lambda key: (key.batch_size, key.mel_frames_bucket),
                    reverse=True,
                )
            )
        self.decoder = decoder
        self.frame_bucket = frame_bucket
        self.n_timesteps = n_timesteps
        self.device = decoder.rand_noise.device
        self.dtype = decoder.rand_noise.dtype
        self.conditioning_dtype = conditioning_dtype
        self.graphs: dict[GraphKey, CapturedFlowGraph] = {}
        self.pool: tuple[int, int] | None = None
        self.capture_attempted = False

    @torch.inference_mode()
    def capture_all(self) -> None:
        """Publish only a complete table; a failed startup build stays eager."""
        if self.capture_attempted:
            raise RuntimeError("Flow graphs may only be captured once at startup")
        else:
            self.capture_attempted = True
        pending: dict[GraphKey, CapturedFlowGraph] = {}
        try:
            with torch.cuda.device(self.device):
                current_stream = torch.cuda.current_stream(self.device)
                stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(current_stream)
                pool = torch.cuda.graph_pool_handle()
                with (
                    torch.cuda.stream(stream),
                    torch.autocast(
                        "cuda",
                        dtype=self.dtype,
                        enabled=self.dtype != torch.float32,
                        cache_enabled=False,
                    ),
                ):
                    for key in self.keys:
                        noise = (
                            self.decoder.rand_noise[..., : key.mel_frames_bucket]
                            .expand(key.batch_size, -1, -1)
                            .clone()
                        )
                        t_span = torch.linspace(
                            0,
                            1,
                            self.n_timesteps + 1,
                            device=self.device,
                            dtype=self.conditioning_dtype,
                        )
                        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
                        mu = torch.zeros_like(noise, dtype=self.conditioning_dtype)
                        mask = torch.ones(
                            key.batch_size,
                            1,
                            key.mel_frames_bucket,
                            device=self.device,
                            dtype=self.conditioning_dtype,
                        )
                        spks = torch.zeros(
                            key.batch_size,
                            self.decoder.out_channels,
                            device=self.device,
                            dtype=self.dtype,
                        )
                        cond = torch.zeros_like(mu)
                        for _ in range(WARMUP_ITERATIONS):
                            self.decoder.solve_euler(
                                noise, t_span, mu, mask, spks, cond
                            )
                        stream.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, pool=pool, stream=stream):
                            output = self.decoder.solve_euler(
                                noise, t_span, mu, mask, spks, cond
                            )
                        pending[key] = CapturedFlowGraph(
                            graph=graph,
                            static_noise=noise,
                            static_t_span=t_span,
                            static_mu=mu,
                            static_mask=mask,
                            static_spks=spks,
                            static_cond=cond,
                            static_output=output,
                        )
                current_stream.wait_stream(stream)
                stream.synchronize()
        except Exception:
            pending.clear()
            logger.exception("MiniCPM-o Flow graph capture failed; using eager solver")
        else:
            self.pool = pool
            self.graphs = pending
            logger.info(f"Captured {len(pending)} MiniCPM-o whole-solver CUDA graphs")

    def select(self, batch_size: int, mel_frames: int) -> GraphKey | None:
        frames = (
            (mel_frames + self.frame_bucket - 1)
            // self.frame_bucket
            * self.frame_bucket
        )
        key = GraphKey(batch_size, frames)
        return key if mel_frames > 0 and key in self.graphs else None

    @torch.inference_mode()
    def run(
        self,
        noise: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return an owned mel tensor; callers must serialize on one CUDA stream."""
        key = self.select(mu.shape[0], mu.shape[-1])
        if key is None:
            return None
        else:
            captured = self.graphs[key]
        frames = mu.shape[-1]
        frame_pairs = (
            (captured.static_noise, noise),
            (captured.static_mu, mu),
            (captured.static_mask, mask),
            (captured.static_cond, cond),
        )
        fixed_pairs = ((captured.static_t_span, t_span), (captured.static_spks, spks))
        if any(
            value.device != static.device
            or value.dtype != static.dtype
            or value.shape != static[..., :frames].shape
            for static, value in frame_pairs
        ) or any(
            value.device != static.device
            or value.dtype != static.dtype
            or value.shape != static.shape
            for static, value in fixed_pairs
        ):
            return None
        else:
            for static, value in frame_pairs:
                static[..., :frames].copy_(value)
                static[..., frames:].zero_()
            for static, value in fixed_pairs:
                static.copy_(value)
            captured.graph.replay()
            return captured.static_output[..., :frames].clone()
