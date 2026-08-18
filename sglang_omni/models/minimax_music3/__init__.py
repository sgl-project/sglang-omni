# SPDX-License-Identifier: Apache-2.0
"""MiniMax Music 3 inference pipeline for SGLang-Omni."""

from sglang_omni.models.model_capabilities import ModelCapabilities

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=False,
    supports_batch_vocoder=False,
    supports_streaming_vocoder=False,
    supports_cuda_graph=True,
    supports_torch_compile=True,
    supports_breakable_prefill_cuda_graph=False,
)

__all__ = ["CAPABILITIES"]
