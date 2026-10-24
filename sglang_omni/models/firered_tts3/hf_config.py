# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hugging Face configuration and variant detection for FireRedTTS3."""

from __future__ import annotations

from typing import Literal

from transformers import AutoConfig, PretrainedConfig

from sglang_omni.utils.json import JsonValue

FIREREDTTS3_MODEL_ARCH_OVERRIDE = "FireRedTTS3ForConditionalGeneration"
FireRedTTS3Variant = Literal["base", "instruct", "speaker"]


class FireRedTTS3Config(PretrainedConfig):
    """Checkpoint fields for the FireRedTTS3 backbone and acoustic components."""

    model_type: str = "fireredtts3"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 40960,
        rms_norm_eps: float = 1e-06,
        rope_theta: float = 1000000.0,
        tie_word_embeddings: bool = True,
        attention_bias: bool = False,
        max_window_layers: int = 28,
        use_sliding_window: bool = False,
        sliding_window: int | None = None,
        bos_token_id: int = 151643,
        eos_token_id: int = 151645,
        redae_dim: int = 64,
        num_history_patches: int = 2,
        patch_size: int = 4,
        patch_encoder_hidden_size: int = 1024,
        patch_encoder_mlp_ratio: int = 4,
        patch_encoder_depth: int = 8,
        patch_encoder_num_heads: int = 16,
        dit_mlp_ratio: int = 3,
        dit_depth: int = 11,
        dit_num_heads: int = 16,
        dit_hidden_size: int = 1024,
        dit_in_channels: int = 1088,
        spk_in_dim: int | None = None,
        audio_patch_size: int = 480,
        audio_sample_rate: int = 24000,
        bottleneck_dim: int = 64,
        enc_hidden_size: int = 896,
        enc_intermediate_size: int = 3584,
        enc_max_position_embeddings: int = 32768,
        enc_max_window_layers: int = 0,
        enc_num_attention_heads: int = 14,
        enc_num_hidden_layers: int = 18,
        enc_num_key_value_heads: int = 2,
        enc_sliding_window: int = 64,
        enc_use_sliding_window: bool = True,
        enc_downsample_num_hidden_layers: int = 4,
        enc_extra_downsample_rate: int = 2,
        dec_hidden_size: int = 896,
        dec_intermediate_size: int = 3584,
        dec_max_position_embeddings: int = 32768,
        dec_max_window_layers: int = 0,
        dec_num_attention_heads: int = 14,
        dec_num_hidden_layers: int = 18,
        dec_num_key_value_heads: int = 2,
        dec_sliding_window: int = 64,
        dec_use_sliding_window: bool = True,
        use_lm_head: bool = False,
        redae_scale: float = 0.4,
        speaker_token: str | None = None,
        text_latent_interleave_ratio: list[int] | None = None,
        special_tokens: dict[str, str | None] | None = None,
        generation_defaults: dict[str, int | float] | None = None,
        **kwargs: JsonValue,
    ) -> None:
        self.vocab_size: int = vocab_size
        self.hidden_size: int = hidden_size
        self.intermediate_size: int = intermediate_size
        self.num_hidden_layers: int = num_hidden_layers
        self.num_attention_heads: int = num_attention_heads
        self.num_key_value_heads: int = num_key_value_heads
        self.head_dim: int = head_dim
        self.hidden_act: str = hidden_act
        self.max_position_embeddings: int = max_position_embeddings
        self.rms_norm_eps: float = rms_norm_eps
        self.rope_theta: float = rope_theta
        self.tie_word_embeddings: bool = tie_word_embeddings
        self.attention_bias: bool = attention_bias
        self.max_window_layers: int = max_window_layers
        self.use_sliding_window: bool = use_sliding_window
        self.sliding_window: int | None = sliding_window
        self.redae_dim: int = redae_dim
        self.num_history_patches: int = num_history_patches
        self.patch_size: int = patch_size
        self.patch_encoder_hidden_size: int = patch_encoder_hidden_size
        self.patch_encoder_mlp_ratio: int = patch_encoder_mlp_ratio
        self.patch_encoder_depth: int = patch_encoder_depth
        self.patch_encoder_num_heads: int = patch_encoder_num_heads
        self.dit_mlp_ratio: int = dit_mlp_ratio
        self.dit_depth: int = dit_depth
        self.dit_num_heads: int = dit_num_heads
        self.dit_hidden_size: int = dit_hidden_size
        self.dit_in_channels: int = dit_in_channels
        self.spk_in_dim: int | None = spk_in_dim
        self.audio_patch_size: int = audio_patch_size
        self.audio_sample_rate: int = audio_sample_rate
        self.bottleneck_dim: int = bottleneck_dim
        self.enc_hidden_size: int = enc_hidden_size
        self.enc_intermediate_size: int = enc_intermediate_size
        self.enc_max_position_embeddings: int = enc_max_position_embeddings
        self.enc_max_window_layers: int = enc_max_window_layers
        self.enc_num_attention_heads: int = enc_num_attention_heads
        self.enc_num_hidden_layers: int = enc_num_hidden_layers
        self.enc_num_key_value_heads: int = enc_num_key_value_heads
        self.enc_sliding_window: int = enc_sliding_window
        self.enc_use_sliding_window: bool = enc_use_sliding_window
        self.enc_downsample_num_hidden_layers: int = enc_downsample_num_hidden_layers
        self.enc_extra_downsample_rate: int = enc_extra_downsample_rate
        self.dec_hidden_size: int = dec_hidden_size
        self.dec_intermediate_size: int = dec_intermediate_size
        self.dec_max_position_embeddings: int = dec_max_position_embeddings
        self.dec_max_window_layers: int = dec_max_window_layers
        self.dec_num_attention_heads: int = dec_num_attention_heads
        self.dec_num_hidden_layers: int = dec_num_hidden_layers
        self.dec_num_key_value_heads: int = dec_num_key_value_heads
        self.dec_sliding_window: int = dec_sliding_window
        self.dec_use_sliding_window: bool = dec_use_sliding_window
        self.use_lm_head: bool = use_lm_head
        self.redae_scale: float = redae_scale
        self.speaker_token: str | None = speaker_token
        self.text_latent_interleave_ratio: list[int] | None = (
            text_latent_interleave_ratio
        )
        self.special_tokens: dict[str, str | None] | None = special_tokens
        self.generation_defaults: dict[str, int | float] | None = generation_defaults
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


def detect_fireredtts3_variant(config: FireRedTTS3Config) -> FireRedTTS3Variant:
    """Classify a checkpoint into its FireRedTTS3 variant from config fields."""
    if config.text_latent_interleave_ratio is not None:
        return "speaker"
    else:
        pass
    if config.spk_in_dim is not None:
        return "base"
    else:
        pass
    if config.use_lm_head:
        return "instruct"
    else:
        pass
    raise ValueError(
        "unrecognized FireRedTTS3 checkpoint: none of the variant markers (text_latent_interleave_ratio / spk_in_dim / use_lm_head) is set"
    )


def register_fireredtts3_hf_config() -> None:
    """Register the local FireRedTTS3 config before SGLang loads ModelConfig."""
    AutoConfig.register("fireredtts3", FireRedTTS3Config, exist_ok=True)


__all__ = [
    "FIREREDTTS3_MODEL_ARCH_OVERRIDE",
    "FireRedTTS3Config",
    "FireRedTTS3Variant",
    "detect_fireredtts3_variant",
    "register_fireredtts3_hf_config",
]
