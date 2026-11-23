# SPDX-License-Identifier: Apache-2.0
# The DiT, SAME decoder, conditioning and ping-pong sampling are adapted from
# Stability-AI/stable-audio-3 (https://github.com/Stability-AI/stable-audio-3)
# at commit 790793b89ddceea612be9ce42ff6830d18f6fa57, licensed under MIT.
"""Stable Audio 3 text-to-audio generation."""

from sglang_omni.models.model_capabilities import ModelCapabilities

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=False,
    supports_batch_vocoder=False,
    supports_streaming_vocoder=False,
    supports_cuda_graph=False,
    supports_torch_compile=False,
    supports_breakable_prefill_cuda_graph=False,
    supports_full_prefill_cuda_graph=False,
)

__all__ = ["CAPABILITIES"]
