# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o 4.5 full-duplex integration."""

from sglang_omni.models.model_capabilities import ModelCapabilities

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=True,
    supports_batch_vocoder=False,
    supports_streaming_vocoder=True,
    supports_cuda_graph=False,
    supports_torch_compile=False,
    supports_native_duplex=True,
    supports_realtime_audio_output=True,
    supports_realtime_video_input=True,
)

__all__ = ["CAPABILITIES"]
