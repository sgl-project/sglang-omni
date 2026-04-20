# SPDX-License-Identifier: Apache-2.0
"""Register sglang-omni model classes in SGLang's ModelRegistry.

Deliberately kept in its own module to avoid importing the heavier
pipeline-config registry (sglang_omni.models.registry) which runs
config discovery at import time.
"""
from __future__ import annotations


def register_omni_models_in_sglang() -> None:
    """Single source of truth — called by both rank-0 ModelRunner and TP followers."""
    from sglang.srt.models.registry import ModelRegistry

    from sglang_omni.models.ming_omni.thinker import (
        BailingMM2Config,
        BailingMoeV2ForCausalLM,
    )

    ModelRegistry.models["BailingMoeV2ForCausalLM"] = BailingMoeV2ForCausalLM

    try:
        from sglang_omni.models.fishaudio_s2_pro.sglang_model import (
            S2ProSGLangTextModel,
        )

        ModelRegistry.models["S2ProSGLangTextModel"] = S2ProSGLangTextModel
    except ImportError:
        pass

    try:
        from sglang_omni.models.qwen3_omni.talker import Qwen3OmniTalker

        ModelRegistry.models["Qwen3OmniTalker"] = Qwen3OmniTalker
    except ImportError:
        pass

    from transformers import AutoConfig

    try:
        AutoConfig.register("bailingmm_moe_v2_lite", BailingMM2Config)
    except ValueError:
        pass  # Already registered
