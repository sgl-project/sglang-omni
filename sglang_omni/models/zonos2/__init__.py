# SPDX-License-Identifier: Apache-2.0
"""ZONOS2 (Zyphra) MoE text-to-speech support for SGLang Omni.

Pipeline: text frontend -> speaker encode -> MoE AR decode -> DAC vocoder (44.1 kHz).
"""

from __future__ import annotations

from sglang_omni.models.model_capabilities import ModelCapabilities

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=True,
    supports_batch_vocoder=True,
    supports_streaming_vocoder=True,
    supports_cuda_graph=True,
    supports_torch_compile=True,
)

__all__ = ["CAPABILITIES"]
