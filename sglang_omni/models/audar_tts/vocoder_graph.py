# SPDX-License-Identifier: Apache-2.0
"""Device graphs for the NeuCodec decoder up to its iSTFT head, one per code count."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import ReplayableGraph

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class DecodeGraph:
    graph: ReplayableGraph
    codes: torch.Tensor
    hidden_states: torch.Tensor

    def replay(self, codes: torch.Tensor) -> torch.Tensor:
        self.codes.copy_(codes)
        self.graph.replay()
        return self.hidden_states


def decode_hidden_states(codec: nn.Module, codes: torch.Tensor) -> torch.Tensor:
    embeddings = codec.generator.quantizer.get_output_from_indices(
        codes.transpose(1, 2)
    )
    return codec.generator.backbone(codec.fc_post_a(embeddings))


def capture_decode_graphs(
    codec: nn.Module, device: torch.device, code_counts: Sequence[int]
) -> dict[int, DecodeGraph]:
    """The iSTFT head stays eager: oneMKL's FFT cannot be captured on XPU.

    The backbone attends over every code, so padding a request would change its audio.
    """
    backend = current_platform.get_device_graph_backend(device)
    if backend is None:
        raise ValueError(f"Audar-TTS vocoder graphs need a graph device; got {device}")
    elif any(code_count <= 0 for code_count in code_counts):
        raise ValueError(
            f"Audar-TTS vocoder graph code counts must be positive; got {code_counts}"
        )
    else:
        pass
    device_module = torch.get_device_module(device)
    graphs: dict[int, DecodeGraph] = {}
    started = time.perf_counter()
    with (
        device_module.device(device),
        torch.inference_mode(),
        current_platform.graph_capture_attention(),
    ):
        stream = device_module.Stream(device=device)
        pool = backend.graph_pool_handle()
        # Largest first, so one shared pool is sized once.
        for code_count in sorted(set(code_counts), reverse=True):
            codes = torch.zeros((1, 1, code_count), dtype=torch.long, device=device)
            stream.wait_stream(device_module.current_stream(device))
            with device_module.stream(stream):
                decode_hidden_states(codec, codes)
            device_module.current_stream(device).wait_stream(stream)
            with backend.capture(pool=pool, stream=stream) as graph:
                hidden_states = decode_hidden_states(codec, codes)
            graphs[code_count] = DecodeGraph(
                graph=graph, codes=codes, hidden_states=hidden_states
            )
        device_module.current_stream(device).wait_stream(stream)
    logger.info(
        f"Audar-TTS vocoder graphs for code counts {sorted(graphs)} captured in "
        f"{time.perf_counter() - started:.2f}s"
    )
    return graphs
