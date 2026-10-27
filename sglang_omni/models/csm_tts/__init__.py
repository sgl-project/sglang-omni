# SPDX-License-Identifier: Apache-2.0
"""Sesame CSM-1B TTS model support for sglang-omni.

Registers CsmTtsHfConfig with transformers.AutoConfig on import so
AutoConfig.from_pretrained() works before any CSM stage factory runs.
The model class is registered in SGLModelRunner.register_omni_model.
"""

from __future__ import annotations

from transformers import AutoConfig

from sglang_omni.models.model_capabilities import ModelCapabilities

from . import config
from .hf_config import CsmTtsHfConfig

AutoConfig.register("csm", CsmTtsHfConfig, exist_ok=True)

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=True,
    supports_batch_vocoder=True,
    supports_streaming_vocoder=True,
    supports_cuda_graph=False,
    supports_torch_compile=False,
    supports_breakable_prefill_cuda_graph=False,
)

__all__ = ["CAPABILITIES", "config", "CsmTtsHfConfig"]
