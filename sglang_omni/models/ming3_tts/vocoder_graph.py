# SPDX-License-Identifier: Apache-2.0
"""Exact-length CUDA graphs for non-streaming Ming vocoding."""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType

import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from sglang_omni.modules.ming_audio.flow.normalization import (
    HAS_TRITON,
    NATIVE_REDUCTION_VECTOR_WIDTH,
    VOCODER_RMS_CHANNEL_COUNT,
    normalize_native_strided_rms,
)
from sglang_omni.modules.ming_audio.vae.modeling_audio_vae import AudioVAE

VOCODER_GRAPH_WARMUP_STEPS = 2


def normalize_vocoder_channels(
    layer: Qwen2RMSNorm, hidden_states: torch.Tensor
) -> torch.Tensor:
    if (
        not torch.is_grad_enabled()
        and hidden_states.device.type == "cuda"
        and hidden_states.dtype in (torch.bfloat16, torch.float16)
        and hidden_states.dtype == layer.weight.dtype
        and hidden_states.ndim == 3
        and hidden_states.shape[0] == 1
        and hidden_states.shape[1] > 0
        and hidden_states.shape[1] % NATIVE_REDUCTION_VECTOR_WIDTH == 0
        and hidden_states.shape[2] == VOCODER_RMS_CHANNEL_COUNT
        and hidden_states.stride(1) == 1
        and hidden_states.stride(2) == hidden_states.shape[1]
        and layer.weight.shape == (VOCODER_RMS_CHANNEL_COUNT,)
        and layer.weight.is_contiguous()
    ):
        return normalize_native_strided_rms(
            hidden_states, layer.weight, layer.variance_epsilon
        )
    else:
        return Qwen2RMSNorm.forward(layer, hidden_states)


@dataclass(kw_only=True)
class CapturedVocoder:
    graph: torch.cuda.CUDAGraph
    latents: torch.Tensor
    waveform: torch.Tensor


class VocoderGraphRunner:
    """Own graph storage and clone results before another length reuses the pool."""

    def __init__(self, vae: AudioVAE, maximum_decode_steps: int) -> None:
        self.vae: AudioVAE = vae
        self.graphs: dict[int, CapturedVocoder] = {}
        self.capture(maximum_decode_steps)

    @torch.inference_mode()
    def capture(self, maximum_decode_steps: int) -> None:
        memory_pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream(device=self.vae.device)
        normalization_layers: list[Qwen2RMSNorm] = []
        try:
            # note (Zhuofeng Li): The fused reduction follows the native CUDA tree in this PyTorch version.
            if (
                HAS_TRITON
                and torch.__version__.split("+")[0] == "2.14.1"
                and torch.version.hip is None
            ):
                for layer in self.vae.decoder.modules():
                    if type(layer) is Qwen2RMSNorm and "forward" not in layer.__dict__:
                        layer.forward = MethodType(normalize_vocoder_channels, layer)
                        normalization_layers.append(layer)
                    else:
                        pass
            else:
                pass
            for latent_steps in range(maximum_decode_steps, 0, -1):
                latents = torch.zeros(
                    (1, latent_steps, self.vae.decoder.latent_dim),
                    device=self.vae.device,
                    dtype=self.vae.dtype,
                )
                stream.wait_stream(torch.cuda.current_stream(self.vae.device))
                with torch.cuda.stream(stream):
                    for _ in range(VOCODER_GRAPH_WARMUP_STEPS):
                        self.vae.decode(latents)
                torch.cuda.current_stream(self.vae.device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=memory_pool, stream=stream):
                    waveform, _, _ = self.vae.decode(latents)
                self.graphs[latent_steps] = CapturedVocoder(
                    graph=graph, latents=latents, waveform=waveform
                )
        finally:
            for layer in normalization_layers:
                layer.__dict__.pop("forward")

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        captured = self.graphs.get(latents.shape[1])
        if captured is None:
            waveform, _, _ = self.vae.decode(latents)
            return waveform
        else:
            captured.latents.copy_(latents)
            captured.graph.replay()
            return captured.waveform.clone()
