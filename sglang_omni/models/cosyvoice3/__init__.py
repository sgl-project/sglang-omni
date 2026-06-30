# SPDX-License-Identifier: Apache-2.0
"""CosyVoice3 TTS model package for sglang-omni.

Vendors CosyVoice (`cosyvoice/`) and Matcha-TTS (`matcha/`) model code; the
sglang-omni glue (config/sglang_model/stages/...) wires it into the 3-stage
preprocessing -> tts_engine -> vocoder pipeline.
"""

from sglang_omni.models.model_capabilities import ModelCapabilities

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=True,
    # The flow (CausalMaskedDiffWithDiT) asserts batch == 1, so the vocoder
    # stage is a batch-1 SimpleScheduler; streaming is a planned increment.
    supports_batch_vocoder=False,
    supports_streaming_vocoder=False,
    supports_cuda_graph=True,
    # No owned torch.compile path (codec/codebook/frame-sampler compiles).
    supports_torch_compile=False,
)

__all__ = ["CAPABILITIES"]
