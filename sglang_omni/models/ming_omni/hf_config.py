# SPDX-License-Identifier: Apache-2.0
"""HuggingFace configuration models for Ming-Omni components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WhisperEncoderConfig:
    """Whisper audio encoder configuration."""

    n_mels: int = 128
    n_ctx: int = 15000
    n_state: int = 1280
    n_head: int = 20
    n_layer: int = 32


@dataclass(frozen=True)
class AudioConfig:
    """Audio processing configuration."""

    whisper_encoder_config: WhisperEncoderConfig = field(
        default_factory=WhisperEncoderConfig
    )
    ds_kernel_size: int = 3
    ds_stride: int = 2
    norm_query_embeds: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AudioConfig":
        wec = d.get("whisper_encoder_config", {})
        return cls(
            whisper_encoder_config=WhisperEncoderConfig(**wec),
            ds_kernel_size=d.get("ds_kernel_size", 3),
            ds_stride=d.get("ds_stride", 2),
            norm_query_embeds=d.get("norm_query_embeds", True),
        )


@dataclass(frozen=True)
class VisionConfig:
    """Vision encoder configuration (Qwen3-style ViT)."""

    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 4096
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: list[int] = field(default_factory=lambda: [8, 16, 24])
    initializer_range: float = 0.02

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VisionConfig":
        return cls(
            depth=d.get("depth", 27),
            hidden_size=d.get("hidden_size", 1152),
            hidden_act=d.get("hidden_act", "gelu_pytorch_tanh"),
            intermediate_size=d.get("intermediate_size", 4304),
            num_heads=d.get("num_heads", 16),
            in_channels=d.get("in_channels", 3),
            patch_size=d.get("patch_size", 16),
            spatial_merge_size=d.get("spatial_merge_size", 2),
            temporal_patch_size=d.get("temporal_patch_size", 2),
            out_hidden_size=d.get("out_hidden_size", 4096),
            num_position_embeddings=d.get("num_position_embeddings", 2304),
            deepstack_visual_indexes=d.get("deepstack_visual_indexes", [8, 16, 24]),
            initializer_range=d.get("initializer_range", 0.02),
        )


@dataclass(frozen=True)
class BailingMoeV2LLMConfig:
    """BailingMoeV2 LLM backbone configuration."""

    vocab_size: int = 157184
    hidden_size: int = 4096
    intermediate_size: int = 9216
    moe_intermediate_size: int = 1024
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 32768
    rope_theta: float = 2400000.0
    partial_rotary_factor: float = 0.5
    use_qk_norm: bool = True
    use_qkv_bias: bool = False
    use_bias: bool = False
    # MoE
    num_experts: int = 256
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    n_group: int = 8
    topk_group: int = 4
    router_type: str = "MultiRouter"
    routed_scaling_factor: float = 2.5
    use_expert_bias: bool = True
    first_k_dense_replace: int = 1
    # Token IDs for multimodal
    eos_token_id: int = 156895
    pad_token_id: int = 156892
    image_patch_token: int = 157157
    video_patch_token: int = 157175
    image_start_token: int = 157158
    video_start_token: int = 157159
    # RoPE scaling
    rope_scaling_type: str = "video_rope"
    use_interleaved_frame_timestamp: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BailingMoeV2LLMConfig":
        rope_scaling = d.get("rope_scaling", {})
        return cls(
            vocab_size=d.get("vocab_size", 157184),
            hidden_size=d.get("hidden_size", 4096),
            intermediate_size=d.get("intermediate_size", 9216),
            moe_intermediate_size=d.get("moe_intermediate_size", 1024),
            num_hidden_layers=d.get("num_hidden_layers", 32),
            num_attention_heads=d.get("num_attention_heads", 32),
            num_key_value_heads=d.get("num_key_value_heads", 4),
            hidden_act=d.get("hidden_act", "silu"),
            rms_norm_eps=d.get("rms_norm_eps", 1e-6),
            max_position_embeddings=d.get("max_position_embeddings", 32768),
            rope_theta=d.get("rope_theta", 2400000.0),
            partial_rotary_factor=d.get("partial_rotary_factor", 0.5),
            use_qk_norm=d.get("use_qk_norm", True),
            use_qkv_bias=d.get("use_qkv_bias", False),
            use_bias=d.get("use_bias", False),
            num_experts=d.get("num_experts", 256),
            num_experts_per_tok=d.get("num_experts_per_tok", 8),
            num_shared_experts=d.get("num_shared_experts", 1),
            n_group=d.get("n_group", 8),
            topk_group=d.get("topk_group", 4),
            router_type=d.get("router_type", "MultiRouter"),
            routed_scaling_factor=d.get(
                "routed_scaling_factor",
                d.get("moe_router_topk_scaling_factor", 2.5),
            ),
            use_expert_bias=d.get("use_expert_bias", True),
            first_k_dense_replace=d.get("first_k_dense_replace", 1),
            eos_token_id=d.get("eos_token_id", 156895),
            pad_token_id=d.get("pad_token_id", 156892),
            image_patch_token=d.get("image_patch_token", 157157),
            video_patch_token=d.get("video_patch_token", 157175),
            image_start_token=d.get("image_start_token", 157158),
            video_start_token=d.get("video_start_token", 157159),
            rope_scaling_type=(
                rope_scaling.get("type", "video_rope") if rope_scaling else "video_rope"
            ),
            use_interleaved_frame_timestamp=d.get(
                "use_interleaved_frame_timestamp", True
            ),
        )


@dataclass(frozen=True)
class MingOmniConfig:
    """Top-level Ming-Omni model configuration."""

    llm_config: BailingMoeV2LLMConfig = field(default_factory=BailingMoeV2LLMConfig)
    audio_config: AudioConfig = field(default_factory=AudioConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    mlp_depth: int = 2
    architecture: str = "BailingMM2NativeForConditionalGeneration"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MingOmniConfig":
        llm_config = BailingMoeV2LLMConfig.from_dict(d.get("llm_config", {}))
        audio_config = AudioConfig.from_dict(d.get("audio_config", {}))
        vision_raw = d.get("vision_config", {})
        vision_config = (
            VisionConfig.from_dict(vision_raw) if vision_raw else VisionConfig()
        )
        return cls(
            llm_config=llm_config,
            audio_config=audio_config,
            vision_config=vision_config,
            mlp_depth=d.get("mlp_depth", 2),
        )
