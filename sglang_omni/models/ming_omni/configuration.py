# SPDX-License-Identifier: Apache-2.0
"""Lightweight Hugging Face configs for Ming-Omni."""

from __future__ import annotations

from transformers import PretrainedConfig


class BailingMoeV2Config(PretrainedConfig):
    """Adapter config that maps Ming's config.json to SGLang expectations."""

    model_type = "bailing_moe_v2"

    def __init__(
        self,
        vocab_size=157184,
        hidden_size=4096,
        intermediate_size=9216,
        moe_intermediate_size=1024,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        max_position_embeddings=32768,
        rope_theta=2400000.0,
        partial_rotary_factor=0.5,
        use_qk_norm=True,
        use_qkv_bias=False,
        num_experts=256,
        num_experts_per_tok=8,
        num_shared_experts=1,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        use_expert_bias=True,
        first_k_dense_replace=1,
        rope_scaling=None,
        tie_word_embeddings=False,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.use_qk_norm = use_qk_norm
        self.use_qkv_bias = use_qkv_bias
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.use_expert_bias = use_expert_bias
        self.first_k_dense_replace = first_k_dense_replace
        if isinstance(rope_scaling, dict) and rope_scaling.get("factor") is None:
            self.rope_scaling = None
        else:
            self.rope_scaling = rope_scaling

        self.head_dim = hidden_size // num_attention_heads
        self.rotary_dim = int(self.head_dim * partial_rotary_factor)


class BailingMM2Config(PretrainedConfig):
    """Top-level composite config for BailingMM2 (Ming-Omni)."""

    model_type = "bailingmm_moe_v2_lite"

    def __init__(
        self,
        mlp_depth=1,
        llm_config=None,
        vision_config=None,
        audio_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mlp_depth = mlp_depth
        self.llm_config = (
            BailingMoeV2Config(**llm_config)
            if isinstance(llm_config, dict)
            else llm_config
        )
        self.audio_config = (
            PretrainedConfig(**audio_config)
            if isinstance(audio_config, dict)
            else audio_config
        )
        self.vision_config = (
            PretrainedConfig(**vision_config)
            if isinstance(vision_config, dict)
            else vision_config
        )
