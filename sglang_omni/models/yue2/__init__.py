# SPDX-License-Identifier: Apache-2.0
# The model equations, checkpoint tensor names, text tokenizer protocol, sampling
# arithmetic and acoustic solver are adapted from YuE2
# (https://github.com/multimodal-art-projection/YuE) at commit
# 1116bea76032bee5668e5f097294d082fa0b84f0, licensed under Apache-2.0.
"""YuE2 music generation."""

from sglang_omni.models.model_capabilities import ModelCapabilities

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=False,
    supports_batch_vocoder=False,
    supports_streaming_vocoder=False,
    supports_cuda_graph=True,
    supports_torch_compile=False,
    supports_breakable_prefill_cuda_graph=False,
    supports_full_prefill_cuda_graph=False,
)

__all__ = ["CAPABILITIES"]
