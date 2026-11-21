# SPDX-License-Identifier: Apache-2.0
"""Lazy registration helpers."""

from __future__ import annotations

import importlib.util


def register_ming_hf_config() -> None:
    """Register Ming's composite HF config before SGLang loads ModelConfig."""
    # note (ratish): SGLang lists its own class for this model type and reloads
    # a checkpoint of that type through its registry, after registering the
    # class with AutoConfig at import. The thinker reads this package's config
    # fields, so both registrations are replaced, on every call.
    from transformers import AutoConfig

    from sglang_omni.models.ming_omni.configuration import BailingMM2Config

    AutoConfig.register(BailingMM2Config.model_type, BailingMM2Config, exist_ok=True)
    # note (WinnieSmasher): the package has to stay importable when sglang is not
    # installed, so the preprocessing path can be exercised without it. Every
    # deployment that loads a Ming checkpoint has sglang, and only then does this
    # registry entry matter. Probing the module instead of catching ImportError
    # keeps a broken sglang install loud rather than silently unregistered.
    if importlib.util.find_spec("sglang") is not None:
        from sglang.srt.utils.hf_transformers import common as sglang_hf_configs

        sglang_hf_configs._CONFIG_REGISTRY[  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            BailingMM2Config.model_type
        ] = BailingMM2Config
    else:
        pass


def register_ming_model_registry() -> None:
    from sglang.srt.models.registry import ModelRegistry

    from sglang_omni.models.ming_omni.thinker import BailingMoeV2ForCausalLM

    ModelRegistry.models["BailingMoeV2ForCausalLM"] = BailingMoeV2ForCausalLM
