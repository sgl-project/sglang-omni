# SPDX-License-Identifier: Apache-2.0
"""Nemotron 3 speaker diarization."""

from sglang_omni.models.model_capabilities import ModelCapabilities

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=False,
    supports_batch_vocoder=False,
    supports_streaming_vocoder=False,
    supports_cuda_graph=False,
    supports_torch_compile=False,
    supports_breakable_prefill_cuda_graph=False,
)
