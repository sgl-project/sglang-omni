# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni-TTS 16B support for SGLang Omni."""

from sglang_omni.models.model_capabilities import ModelCapabilities

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=True,
    supports_batch_vocoder=True,
    supports_streaming_vocoder=False,
    supports_cuda_graph=True,
    supports_torch_compile=False,
)

__all__ = ["CAPABILITIES"]
