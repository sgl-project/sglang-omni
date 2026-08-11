# SPDX-License-Identifier: Apache-2.0
"""Framework-native dots.tts support for sglang-omni."""

from sglang_omni.models.model_capabilities import ModelCapabilities

from . import config

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=True,
    supports_batch_vocoder=True,
    supports_streaming_vocoder=True,
    supports_cuda_graph=False,
    supports_torch_compile=True,
    supports_breakable_prefill_cuda_graph=False,
)

__all__ = ["CAPABILITIES", "config"]
