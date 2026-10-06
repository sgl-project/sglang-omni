# SPDX-License-Identifier: Apache-2.0
"""CUDA graph capture for fixed-shape VoiceChat kernels."""

from typing import Protocol

import torch


class GraphForward(Protocol):
    def __call__(self) -> torch.Tensor: ...


@torch.inference_mode()
def capture_cuda_graph(
    forward: GraphForward,
    device: torch.device,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(capture_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        output = forward()
    return graph, output
