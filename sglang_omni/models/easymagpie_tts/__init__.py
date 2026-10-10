# SPDX-License-Identifier: Apache-2.0
"""EasyMagpie text-to-speech support for SGLang Omni.

Pipeline: text frontend -> Nemotron-H AR talker -> causal FSQ codec (22.05 kHz).
"""

from __future__ import annotations

from sglang_omni.models.model_capabilities import ModelCapabilities

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=False,
    supports_batch_vocoder=True,
    supports_streaming_vocoder=True,
    supports_cuda_graph=True,
    supports_torch_compile=False,
    supports_breakable_prefill_cuda_graph=True,
)

__all__ = ["CAPABILITIES"]
