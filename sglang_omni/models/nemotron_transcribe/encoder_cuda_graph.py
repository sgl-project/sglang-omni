# SPDX-License-Identifier: Apache-2.0
"""Bounded encoder CUDA graphs, independent of the decoder's capture pool."""

from dataclasses import dataclass
from typing import Protocol

import torch

MAX_CACHED_ENCODER_SHAPES = 4
ENCODER_GRAPH_WARMUP_STEPS = 2


class EncoderLayersForward(Protocol):
    def __call__(
        self, hidden_states: torch.Tensor, valid_key_mask: torch.Tensor | None, /
    ) -> torch.Tensor: ...


@dataclass(frozen=True, kw_only=True)
class EncoderGraphShape:
    hidden_shape: tuple[int, ...]
    hidden_strides: tuple[int, ...]
    hidden_dtype: torch.dtype
    device: torch.device
    mask_shape: tuple[int, ...] | None
    mask_strides: tuple[int, ...] | None


@dataclass(kw_only=True)
class EncoderGraph:
    graph: torch.cuda.CUDAGraph
    hidden_states: torch.Tensor
    valid_key_mask: torch.Tensor | None
    output: torch.Tensor


class EncoderCudaGraphs:
    def __init__(self, forward: EncoderLayersForward) -> None:
        self.forward = forward
        self.captures: dict[EncoderGraphShape, EncoderGraph] = {}

    def __call__(
        self, hidden_states: torch.Tensor, valid_key_mask: torch.Tensor | None
    ) -> torch.Tensor:
        shape = EncoderGraphShape(
            hidden_shape=tuple(hidden_states.shape),
            hidden_strides=hidden_states.stride(),
            hidden_dtype=hidden_states.dtype,
            device=hidden_states.device,
            mask_shape=None if valid_key_mask is None else tuple(valid_key_mask.shape),
            mask_strides=None if valid_key_mask is None else valid_key_mask.stride(),
        )
        capture = self.captures.get(shape)
        if capture is None:
            if len(self.captures) >= MAX_CACHED_ENCODER_SHAPES:
                return self.forward(hidden_states, valid_key_mask)
            else:
                pass
            static_hidden_states = hidden_states.clone()
            static_key_mask = None if valid_key_mask is None else valid_key_mask.clone()
            current_stream = torch.cuda.current_stream(hidden_states.device)
            capture_stream = torch.cuda.Stream(device=hidden_states.device)
            capture_stream.wait_stream(current_stream)
            with torch.cuda.stream(capture_stream):
                for _ in range(ENCODER_GRAPH_WARMUP_STEPS):
                    self.forward(static_hidden_states, static_key_mask)
            current_stream.wait_stream(capture_stream)
            graph = torch.cuda.CUDAGraph()
            # note (yhayaran): Separate pools allow shapes to replay in any order.
            with torch.cuda.graph(graph, stream=capture_stream):
                static_output = self.forward(static_hidden_states, static_key_mask)
            current_stream.wait_stream(capture_stream)
            capture = EncoderGraph(
                graph=graph,
                hidden_states=static_hidden_states,
                valid_key_mask=static_key_mask,
                output=static_output,
            )
            self.captures[shape] = capture
        else:
            pass
        capture.hidden_states.copy_(hidden_states)
        if capture.valid_key_mask is not None:
            assert valid_key_mask is not None
            capture.valid_key_mask.copy_(valid_key_mask)
        else:
            pass
        capture.graph.replay()
        # note (yhayaran): Later windows and requests must not overwrite results.
        return capture.output.clone()
