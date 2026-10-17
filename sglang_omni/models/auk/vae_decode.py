# SPDX-License-Identifier: Apache-2.0
"""Exact-length graph replay for the AuK waveform decoder."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import NamedTuple

import torch

from sglang_omni.models.auk.vae import BigVGANFlowVAE
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import ReplayableGraph

logger = logging.getLogger(__name__)


class CapturedVaeDecode(NamedTuple):
    graph: ReplayableGraph
    latents: torch.Tensor
    waveforms: torch.Tensor


class AuKVaeDecoder:
    """Decode declared shapes with graphs and other shapes eagerly."""

    @torch.inference_mode()
    def __init__(
        self,
        vae: BigVGANFlowVAE,
        device: torch.device,
        *,
        capture_shapes: Sequence[Sequence[int]],
        compile_forward: bool,
    ) -> None:
        self.vae = vae
        self.graphs: dict[tuple[int, int], CapturedVaeDecode] = {}
        shapes: set[tuple[int, int]] = set()
        for shape in capture_shapes:
            if len(shape) != 2 or any(dimension < 1 for dimension in shape):
                raise ValueError(
                    "AuK VAE graph shapes need positive batch and frame counts"
                )
            else:
                shapes.add((shape[0], shape[1]))
        backend = current_platform.get_device_graph_backend(device)
        if not shapes or backend is None:
            return
        else:
            pass
        operation = (
            torch.compile(
                self.decode_eager,
                fullgraph=True,
                dynamic=True,
                options={"triton.cudagraphs": False},
            )
            if compile_forward
            else self.decode_eager
        )
        module = torch.get_device_module(device)
        stream = module.Stream(device=device)
        pool = module.graph_pool_handle()
        for batch_size, frames in sorted(shapes):
            latents = torch.zeros(
                (batch_size, frames, vae.h.latent_dim),
                device=device,
                dtype=torch.float32,
            )
            stream.wait_stream(module.current_stream(device))
            with module.stream(stream):
                for _ in range(3):
                    operation(latents)
                stream.synchronize()
                with backend.capture(pool=pool, stream=stream) as graph:
                    waveforms = operation(latents)
            self.graphs[batch_size, frames] = CapturedVaeDecode(
                graph, latents, waveforms
            )
        module.current_stream(device).wait_stream(stream)
        module.synchronize(device)
        logger.info(f"AuK VAE: captured {len(self.graphs)} exact decode shapes")

    def decode_eager(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.inference_from_latents(
            self.vae.denormalize(latents).permute(0, 2, 1)
        )

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Return waveforms to consume before the next replay reuses its buffers."""
        captured = self.graphs.get((latents.shape[0], latents.shape[1]))
        if captured is None:
            return self.decode_eager(latents)
        else:
            captured.latents.copy_(latents)
            captured.graph.replay()
            return captured.waveforms
