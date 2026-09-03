# SPDX-License-Identifier: Apache-2.0
"""HiggsMultimodalQwen3 TTS model support for sglang-omni.

Registers :class:`HiggsMultimodalQwen3Config` with ``transformers.AutoConfig`` on
import so ``AutoConfig.from_pretrained()`` works before any Higgs stage factory
runs. The model class is registered in
:meth:`sglang_omni.model_runner.sglang_model_runner.SGLModelRunner._register_omni_model`
alongside the other sglang-omni models.
"""

from __future__ import annotations

from transformers import AutoConfig

from sglang_omni.models.model_capabilities import ModelCapabilities
from sglang_omni.platforms import current_platform

from . import config
from .hf_config import HiggsMultimodalQwen3Config

AutoConfig.register("higgs_multimodal_qwen3", HiggsMultimodalQwen3Config)

# NPU's compile backend cannot compile this codec model 
# (crashes on dynamic shapes, verified on Atlas 910B).
_supports_torch_compile = not current_platform.is_npu()

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=True,
    supports_batch_vocoder=True,
    supports_streaming_vocoder=True,
    supports_cuda_graph=True,
    supports_torch_compile=_supports_torch_compile,
    supports_breakable_prefill_cuda_graph=True,
)

__all__ = ["CAPABILITIES", "config", "HiggsMultimodalQwen3Config"]
