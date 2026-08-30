# SPDX-License-Identifier: Apache-2.0
"""Runtime registration for VoiceChat's SGLang-backed stages."""

from __future__ import annotations

_registered = False


def register_voicechat_models() -> None:
    """Register VoiceChat configs and models before SGLang builds an engine."""
    global _registered
    if _registered:
        return

    from sglang.srt.models.registry import ModelRegistry
    from transformers import AutoConfig

    from .configuration import EarTTSConfig
    from .talker import EarTTSForCausalLM
    from .thinker import NemotronDuplexHForCausalLM

    AutoConfig.register(EarTTSConfig.model_type, EarTTSConfig, exist_ok=True)
    ModelRegistry.models["NemotronDuplexHForCausalLM"] = NemotronDuplexHForCausalLM
    ModelRegistry.models["EarTTSForCausalLM"] = EarTTSForCausalLM
    _registered = True


__all__ = ["register_voicechat_models"]
