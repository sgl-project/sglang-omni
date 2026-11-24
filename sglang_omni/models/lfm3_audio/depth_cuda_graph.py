# SPDX-License-Identifier: Apache-2.0
"""CUDA graphs of LFM3's per-frame depth work, one per decode batch bucket.

The exact sampler replays one depth pass (hidden states, codes and mask to logits) per graph; its draws stay eager on each request's generator. The seeded sampler replays the whole frame: three depth passes, the repetition penalty, top-k and the hashed draws.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from sglang_omni.models.lfm3_audio.depth import Lfm3DepthDecoder, seeded_frame_codes
from sglang_omni.platforms.device_graph import (
    DeviceGraphBackend,
    DeviceGraphPool,
    ReplayableGraph,
)

logger = logging.getLogger(__name__)

WARMUP_ITERATIONS = 2


class GraphCompute(Protocol):
    """The work one graph records, reading its static input buffers."""

    def __call__(self) -> torch.Tensor: ...


@dataclass(kw_only=True)
class DepthPassGraph:
    """Static buffers and graph of one bucket of depth passes."""

    graph: ReplayableGraph
    hidden_states: torch.Tensor
    codes: torch.Tensor
    masked: torch.Tensor
    logits: torch.Tensor


@dataclass(kw_only=True)
class SeededFrameInputs:
    """Static input buffers of one bucket of whole seeded frames."""

    hidden_states: torch.Tensor
    seeds: torch.Tensor
    frame_positions: torch.Tensor
    temperature: torch.Tensor
    top_k: torch.Tensor
    repetition_penalty: torch.Tensor
    history: torch.Tensor


@dataclass(kw_only=True)
class SeededFrameGraph:
    """Input buffers, graph and output codes of one bucket of whole seeded frames."""

    graph: ReplayableGraph
    inputs: SeededFrameInputs
    codes: torch.Tensor


class Lfm3DepthCudaGraphs:
    """Captures and replays the depth graphs; a batch above the largest bucket runs eagerly."""

    def __init__(
        self,
        decoder: Lfm3DepthDecoder,
        graph_backend: DeviceGraphBackend,
        *,
        hidden_size: int,
        schedule: Sequence[int],
        repetition_window: int,
    ) -> None:
        self.decoder = decoder
        self.graph_backend = graph_backend
        self.hidden_size = hidden_size
        self.schedule = list(schedule)
        self.repetition_window = repetition_window
        self.num_codebooks = sum(schedule)
        reference = decoder.mask_embedding
        self.device = reference.device
        self.dtype = decoder.depth_linear.weight.dtype
        self.pool: DeviceGraphPool | None = None
        self.pass_graphs: dict[int, DepthPassGraph] = {}
        self.frame_graphs: dict[int, SeededFrameGraph] = {}

    def warm_up_and_capture(
        self, compute: GraphCompute
    ) -> tuple[ReplayableGraph, torch.Tensor]:
        """The captured graph and the static tensor its replays write."""
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(WARMUP_ITERATIONS):
                compute()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        with self.graph_backend.capture(
            pool=self.pool, thread_local_errors=True
        ) as graph:
            output = compute()
        return graph, output

    @torch.no_grad()
    def capture_pass_graphs(self, batch_sizes: Sequence[int]) -> None:
        self.pool = self.graph_backend.graph_pool_handle()
        for bucket in sorted({int(size) for size in batch_sizes}, reverse=True):
            hidden_states = torch.zeros(
                bucket, self.hidden_size, device=self.device, dtype=self.dtype
            )
            codes = torch.zeros(
                bucket, self.num_codebooks, device=self.device, dtype=torch.long
            )
            masked = torch.ones_like(codes, dtype=torch.bool)
            graph, logits = self.warm_up_and_capture(
                PassCompute(self.decoder, hidden_states, codes, masked)
            )
            self.pass_graphs[bucket] = DepthPassGraph(
                graph=graph,
                hidden_states=hidden_states,
                codes=codes,
                masked=masked,
                logits=logits,
            )
        logger.info(
            f"LFM3 depth pass CUDA graphs captured for bs={sorted(self.pass_graphs)}"
        )

    @torch.no_grad()
    def capture_frame_graphs(self, batch_sizes: Sequence[int]) -> None:
        self.pool = self.graph_backend.graph_pool_handle()
        for bucket in sorted({int(size) for size in batch_sizes}, reverse=True):
            inputs = SeededFrameInputs(
                hidden_states=torch.zeros(
                    bucket, self.hidden_size, device=self.device, dtype=self.dtype
                ),
                seeds=torch.zeros(bucket, device=self.device, dtype=torch.long),
                frame_positions=torch.zeros(
                    bucket, device=self.device, dtype=torch.long
                ),
                temperature=torch.ones(bucket, device=self.device),
                top_k=torch.zeros(bucket, device=self.device, dtype=torch.long),
                repetition_penalty=torch.ones(bucket, device=self.device),
                history=torch.zeros(
                    bucket,
                    self.repetition_window,
                    self.num_codebooks,
                    device=self.device,
                    dtype=torch.long,
                ),
            )
            graph, codes = self.warm_up_and_capture(
                FrameCompute(self.decoder, inputs, self.schedule)
            )
            self.frame_graphs[bucket] = SeededFrameGraph(
                graph=graph, inputs=inputs, codes=codes
            )
        logger.info(
            f"LFM3 seeded frame CUDA graphs captured for bs={sorted(self.frame_graphs)}"
        )

    def depth_logits(
        self, hidden_states: torch.Tensor, codes: torch.Tensor, masked: torch.Tensor
    ) -> torch.Tensor:
        rows = hidden_states.shape[0]
        bucket = min((size for size in self.pass_graphs if size >= rows), default=None)
        if bucket is None:
            return self.decoder(hidden_states, codes, masked)
        else:
            pass
        pass_graph = self.pass_graphs[bucket]
        pass_graph.hidden_states[:rows].copy_(hidden_states)
        pass_graph.codes[:rows].copy_(codes)
        pass_graph.masked[:rows].copy_(masked)
        pass_graph.graph.replay()
        # note (ruilin): the caller consumes these logits before the next replay on the same stream.
        return pass_graph.logits[:rows]

    def seeded_frames(
        self,
        hidden_states: torch.Tensor,
        seeds: torch.Tensor,
        frame_positions: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        repetition_penalty: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        rows = hidden_states.shape[0]
        bucket = min((size for size in self.frame_graphs if size >= rows), default=None)
        if bucket is None:
            return seeded_frame_codes(
                self.decoder,
                hidden_states,
                seeds,
                frame_positions,
                temperature,
                top_k,
                repetition_penalty,
                history,
                self.schedule,
            )
        else:
            pass
        frame_graph = self.frame_graphs[bucket]
        inputs = frame_graph.inputs
        inputs.hidden_states[:rows].copy_(hidden_states)
        inputs.seeds[:rows].copy_(seeds)
        inputs.frame_positions[:rows].copy_(frame_positions)
        inputs.temperature[:rows].copy_(temperature)
        inputs.top_k[:rows].copy_(top_k)
        inputs.repetition_penalty[:rows].copy_(repetition_penalty)
        inputs.history[:rows].copy_(history)
        frame_graph.graph.replay()
        return frame_graph.codes[:rows].clone()


class PassCompute:
    def __init__(
        self,
        decoder: Lfm3DepthDecoder,
        hidden_states: torch.Tensor,
        codes: torch.Tensor,
        masked: torch.Tensor,
    ) -> None:
        self.decoder = decoder
        self.hidden_states = hidden_states
        self.codes = codes
        self.masked = masked

    def __call__(self) -> torch.Tensor:
        return self.decoder(self.hidden_states, self.codes, self.masked)


class FrameCompute:
    def __init__(
        self,
        decoder: Lfm3DepthDecoder,
        inputs: SeededFrameInputs,
        schedule: Sequence[int],
    ) -> None:
        self.decoder = decoder
        self.inputs = inputs
        self.schedule = schedule

    def __call__(self) -> torch.Tensor:
        inputs = self.inputs
        return seeded_frame_codes(
            self.decoder,
            inputs.hidden_states,
            inputs.seeds,
            inputs.frame_positions,
            inputs.temperature,
            inputs.top_k,
            inputs.repetition_penalty,
            inputs.history,
            self.schedule,
        )
