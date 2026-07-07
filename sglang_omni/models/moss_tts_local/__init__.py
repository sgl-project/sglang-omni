# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS Local (v1.5) pipeline package."""

from sglang_omni.models.model_capabilities import ModelCapabilities

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=True,
    supports_batch_vocoder=True,
    supports_streaming_vocoder=True,
    supports_cuda_graph=True,
    supports_torch_compile=True,
)

__all__ = ["CAPABILITIES"]
