# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 model support for sglang-omni."""

from sglang_omni.models.model_capabilities import ModelCapabilities

from . import config

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=True,
    supports_batch_vocoder=True,
    # Flow + HiFT currently run after the complete LLM result is available.
    # Keep this false until incremental decoder wiring is implemented.
    supports_streaming_vocoder=False,
    supports_cuda_graph=True,
    supports_torch_compile=True,
)

__all__ = ["CAPABILITIES", "config"]
