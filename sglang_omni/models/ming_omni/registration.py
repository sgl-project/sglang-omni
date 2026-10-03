# SPDX-License-Identifier: Apache-2.0
"""Lazy registration helpers."""

from __future__ import annotations


def register_ming_hf_config() -> None:
    """Register Ming's composite HF config before SGLang loads ModelConfig."""
    # note (ratish): SGLang registers its own class for this model type when
    # its config utilities are first imported, which replaces an earlier
    # registration. Import them first and register on every call.
    import sglang.srt.utils.hf_transformers_utils  # noqa: F401
    from transformers import AutoConfig

    from sglang_omni.models.ming_omni.configuration import BailingMM2Config

    AutoConfig.register("bailingmm_moe_v2_lite", BailingMM2Config, exist_ok=True)


def register_ming_model_registry() -> None:
    from sglang.srt.models.registry import ModelRegistry

    from sglang_omni.models.ming_omni.thinker import BailingMoeV2ForCausalLM

    ModelRegistry.models["BailingMoeV2ForCausalLM"] = BailingMoeV2ForCausalLM
