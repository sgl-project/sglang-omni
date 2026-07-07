"""Voxtral-4B-TTS model support for sglang-omni."""

from sglang_omni.models.model_capabilities import ModelCapabilities

from . import config

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=False,
    supports_batch_vocoder=False,
    supports_streaming_vocoder=False,
    supports_cuda_graph=True,
    supports_torch_compile=True,
)

__all__ = ["CAPABILITIES", "config"]
