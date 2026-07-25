# SPDX-License-Identifier: Apache-2.0
"""Audar-TTS support for SGLang Omni."""

from sglang_omni.models.model_capabilities import ModelCapabilities

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=True,
    supports_batch_vocoder=False,
    supports_streaming_vocoder=False,
    supports_cuda_graph=False,
    supports_torch_compile=False,
)

__all__ = ["CAPABILITIES"]
