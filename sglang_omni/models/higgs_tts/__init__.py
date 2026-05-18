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

from . import config
from .hf_config import HiggsMultimodalQwen3Config

AutoConfig.register("higgs_multimodal_qwen3", HiggsMultimodalQwen3Config)

__all__ = ["config", "HiggsMultimodalQwen3Config"]
