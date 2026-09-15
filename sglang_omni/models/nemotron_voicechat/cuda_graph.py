# SPDX-License-Identifier: Apache-2.0
"""CUDA graph capture for fixed-shape VoiceChat kernels."""

from collections.abc import Callable

import torch


@torch.inference_mode()
def capture_cuda_graph(
    forward: Callable[[], torch.Tensor],
    device: torch.device,
    *,
    restore_state: Callable[[], None] | None = None,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    """Warm kernels before capture, restoring causal state when necessary."""
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
    if restore_state is not None:
        restore_state()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = forward()
    return graph, output
