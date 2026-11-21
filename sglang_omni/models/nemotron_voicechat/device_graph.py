# SPDX-License-Identifier: Apache-2.0
"""Device graph capture for fixed-shape VoiceChat kernels."""

from typing import Protocol

import torch

from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import ReplayableGraph


class GraphForward(Protocol):
    def __call__(self) -> torch.Tensor: ...


@torch.inference_mode()
def capture_voicechat_graph(
    forward: GraphForward,
    device: torch.device,
) -> tuple[ReplayableGraph, torch.Tensor]:
    backend = current_platform.get_device_graph_backend(device)
    if backend is None:
        raise ValueError(f"No VoiceChat graph backend for {device}")
    else:
        pass
    device_module = torch.get_device_module(device)
    capture_stream = device_module.Stream(device=device)
    current_stream = device_module.current_stream(device)
    capture_stream.wait_stream(current_stream)
    with device_module.stream(capture_stream):
        for _ in range(3):
            forward()
    current_stream.wait_stream(capture_stream)
    with backend.capture(
        stream=capture_stream, thread_local_errors=device.type == "npu"
    ) as graph:
        output = forward()
    return graph, output
