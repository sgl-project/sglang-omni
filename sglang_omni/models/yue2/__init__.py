# SPDX-License-Identifier: Apache-2.0
"""YuE2 music generation pipeline for SGLang-Omni.

lyrics + style/tags -> ABC plan -> semantic codec tokens -> NAR acoustic latents
-> 48 kHz stereo audio. The AR-NAR math and the SGLang-YuE2 optimizations
(fused decode graphs, fast sampler, FA3-varlen, batched modules) are ported into
this package; see ``MIGRATION.md``.
"""

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
