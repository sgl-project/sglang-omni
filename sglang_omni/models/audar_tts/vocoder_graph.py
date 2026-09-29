# SPDX-License-Identifier: Apache-2.0
"""Graph replay for the NeuCodec decoder up to its iSTFT head, one graph per code count."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from sglang_omni.platforms import current_platform

logger = logging.getLogger(__name__)


class ReplayableGraph(Protocol):
    def replay(self) -> None: ...


@dataclass(frozen=True, kw_only=True)
class DecodeGraph:
    """A recorded decoder front and the static buffers its replay reads and writes."""

    graph: ReplayableGraph
    codes: torch.Tensor
    hidden: torch.Tensor

    def replay(self, codes: torch.Tensor) -> torch.Tensor:
        """The iSTFT head's input for codes; the next replay overwrites it."""
        self.codes.copy_(codes)
        self.graph.replay()
        return self.hidden


def decode_hidden(codec: nn.Module, codes: torch.Tensor) -> torch.Tensor:
    """NeuCodec.decode_code up to the input of its iSTFT head."""
    embedding = codec.generator.quantizer.get_output_from_indices(codes.transpose(1, 2))
    return codec.generator.backbone(codec.fc_post_a(embedding))


def capture_decode_graphs(
    codec: nn.Module, device: torch.device, code_counts: Iterable[int]
) -> dict[int, DecodeGraph]:
    """Record one decoder-front graph per code count, largest first into one pool.

    The backbone attends over the whole unmasked sequence, so a graph serves
    exactly one code count: padding a shorter request would change its audio.
    """
    backend = current_platform.get_device_graph_backend(device)
    counts = sorted(set(code_counts), reverse=True)
    if backend is None:
        raise ValueError(
            f"Audar-TTS vocoder graphs need a graph-capable device; got {device}"
        )
    elif not counts or counts[-1] < 1:
        raise ValueError(
            f"Audar-TTS vocoder graph code counts must be positive; got {counts}"
        )
    else:
        pass
    module = torch.get_device_module(device)
    stream = module.Stream(device=device)
    pool = module.graph_pool_handle()
    graphs: dict[int, DecodeGraph] = {}
    free_before, _ = module.mem_get_info(device)
    started = time.perf_counter()
    with torch.inference_mode(), current_platform.graph_capture_attention():
        for code_count in counts:
            codes = torch.zeros((1, 1, code_count), dtype=torch.long, device=device)
            stream.wait_stream(module.current_stream(device))
            with module.stream(stream):
                decode_hidden(codec, codes)
            module.current_stream(device).wait_stream(stream)
            with backend.capture(pool=pool, stream=stream) as graph:
                hidden = decode_hidden(codec, codes)
            graphs[code_count] = DecodeGraph(graph=graph, codes=codes, hidden=hidden)
    free_after, _ = module.mem_get_info(device)
    logger.info(
        f"Audar-TTS vocoder graphs: captured code counts {sorted(graphs)} in "
        f"{time.perf_counter() - started:.2f}s "
        f"({(free_before - free_after) / 2**20:.0f}MiB device memory)"
    )
    return graphs
