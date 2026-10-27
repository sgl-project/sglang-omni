# SPDX-License-Identifier: Apache-2.0
"""Exact-(B, T) CUDA graphs over the dots.tts slot-pool AudioVAE step."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)
HIT_RATE_LOG_EVERY = 1024

StepInputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
StepOutputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
StepForward = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    StepOutputs,
]


class CapturedStepGraph(NamedTuple):
    graph: torch.cuda.CUDAGraph
    packed: torch.Tensor
    hidden_h: torch.Tensor
    hidden_c: torch.Tensor
    window: torch.Tensor
    valid: torch.Tensor
    outputs: StepOutputs


class DotsVocoderGraphRunner:
    """Replays one captured graph per (batch, frames) key; misses return None.

    Inputs are copied into static buffers, outputs are cloned out, so callers
    keep the eager step's value semantics and a replay never aliases the
    previous step's result.
    """

    def __init__(
        self,
        *,
        forward: StepForward,
        new_inputs: Callable[[int, int], StepInputs],
        device: torch.device,
        warmup_iters: int = 2,
    ) -> None:
        self.forward = forward
        self.new_inputs = new_inputs
        self.device = device
        self.warmup_iters = int(warmup_iters)
        self.graphs: dict[tuple[int, int], CapturedStepGraph] = {}
        self.replays = 0
        self.misses = 0

    @property
    def captured_keys(self) -> list[tuple[int, int]]:
        return sorted(self.graphs)

    @torch.no_grad()
    def capture(self, keys: Iterable[tuple[int, int]]) -> None:
        if self.device.type != "cuda":
            raise RuntimeError("dots.tts vocoder CUDA graphs require a CUDA device")
        else:
            pass
        # note (lennox): capture largest first so smaller graphs reuse freed blocks.
        ordered = sorted(
            {(int(b), int(t)) for b, t in keys} - set(self.graphs), reverse=True
        )
        with torch.cuda.device(self.device):
            current_stream = torch.cuda.current_stream(self.device)
            capture_stream = torch.cuda.Stream(device=self.device)
            capture_stream.wait_stream(current_stream)
            graph_pool = torch.cuda.graph_pool_handle()
            for key in ordered:
                inputs = self.new_inputs(*key)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(capture_stream):
                    for _ in range(self.warmup_iters):
                        self.forward(*inputs)
                capture_stream.synchronize()
                with torch.cuda.graph(
                    graph,
                    pool=graph_pool,
                    stream=capture_stream,
                    capture_error_mode="thread_local",
                ):
                    outputs = self.forward(*inputs)
                self.graphs[key] = CapturedStepGraph(graph, *inputs, outputs)
            current_stream.wait_stream(capture_stream)
            torch.cuda.synchronize(self.device)
        logger.info(
            f"dots.tts streaming vocoder CUDA graphs captured: {self.captured_keys}"
        )

    @torch.no_grad()
    def run(
        self,
        packed: torch.Tensor,
        hidden_h: torch.Tensor,
        hidden_c: torch.Tensor,
        window: torch.Tensor,
        valid: torch.Tensor,
    ) -> StepOutputs | None:
        captured = self.graphs.get((int(packed.shape[0]), int(packed.shape[2])))
        if captured is None:
            self.misses += 1
            outputs = None
        else:
            captured.packed.copy_(packed)
            captured.hidden_h.copy_(hidden_h)
            captured.hidden_c.copy_(hidden_c)
            captured.window.copy_(window)
            captured.valid.copy_(valid)
            captured.graph.replay()
            self.replays += 1
            outputs = tuple(output.clone() for output in captured.outputs)
        calls = self.replays + self.misses
        milestone = calls == 1 or calls % HIT_RATE_LOG_EVERY == 0
        logger.log(
            logging.INFO if milestone else logging.DEBUG,
            f"dots.tts streaming vocoder CUDA graph replays={self.replays} "
            f"misses={self.misses} ({100.0 * self.replays / calls:.1f}% hit)",
        )
        return outputs


__all__ = ["DotsVocoderGraphRunner"]
