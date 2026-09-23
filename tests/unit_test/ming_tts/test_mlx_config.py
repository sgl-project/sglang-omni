# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from sglang_omni.models.ming_tts.mlx.config import AcousticConfig, ModelConfig, TextConfig


def text_config_dict() -> dict[str, Any]:
    return dict(vocab_size=32, hidden_size=16, intermediate_size=24,
                moe_intermediate_size=12, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                num_experts=4, num_experts_per_tok=2, num_shared_experts=1,
                first_k_dense_replace=1, multi_gate=True,
                rope_scaling={"type": "3D", "factor": None, "mrope_section": [1, 1, 2]})


def test_composite_config_accepts_tiny_a3b_structure_without_importing_mlx() -> None:
    config = ModelConfig.from_dict(dict(
        llm_config=text_config_dict(),
        ditar_config=dict(hidden_size=16, depth=2, num_heads=2, patch_size=2, history_patch_size=4),
        aggregator_config=dict(hidden_size=16, depth=1, num_heads=2),
        audio_tokenizer_config={"enc_kwargs": {"latent_dim": 4}},
        architectures=["BailingMMNativeForConditionalGeneration"],
    ))
    assert config.llm_config.mrope_section == (1, 1, 2)
    assert (config.patch_size, config.history_patch_size, config.latent_dim) == (2, 4, 4)


@pytest.mark.parametrize("change", [
    {"model_type": "qwen2"}, {"num_experts": 0}, {"num_experts_per_tok": 5},
    {"use_qk_norm": True}, {"use_sliding_window": True},
    {"score_function": "sigmoid"}, {"hidden_act": "gelu"},
    {"router_dtype": "float32"}, {"n_group": 2},
    {"moe_router_enable_expert_bias": True},
    {"moe_shared_expert_intermediate_size": 99},
    {"rope_scaling": {"type": "linear", "factor": 2}},
    {"rope_scaling": {"type": "3D", "factor": None}},
])
def test_reject_unsupported_text_variants(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        TextConfig.from_dict({**text_config_dict(), **change})


def test_official_mrope_sections() -> None:
    config = TextConfig.from_dict({**text_config_dict(), "head_dim": 128,
                                  "rope_scaling": {"type": "3D", "factor": None}})
    assert config.mrope_section == (16, 24, 24)


@pytest.mark.parametrize("change", [{"qk_norm": "rms_norm"}, {"pe_attn_head": 1}, {"spk_dim": 192}])
def test_reject_unsupported_acoustic_variants(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        AcousticConfig.from_dict(dict(hidden_size=16, depth=1, num_heads=2, **change))
