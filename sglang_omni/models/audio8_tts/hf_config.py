# SPDX-License-Identifier: Apache-2.0
"""Checkpoint configuration for Audio8 TTS."""

from __future__ import annotations

from pydantic import JsonValue
from transformers import Qwen2Config

SAMPLE_RATE_HZ = 44_100
FRAME_SAMPLES = 2048


class ArkttsConfig(Qwen2Config):
    model_type = "arktts"

    def __init__(
        self,
        vocab_size: int = 155_776,
        dim: int = 896,
        n_layer: int = 24,
        n_head: int = 14,
        n_local_heads: int = 2,
        head_dim: int = 64,
        intermediate_size: int = 4864,
        max_seq_len: int = 2048,
        rope_base: float = 1_000_000,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        attention_qkv_bias: bool = True,
        attention_qk_norm: bool = False,
        attention_o_bias: bool = False,
        tie_word_embeddings: bool = True,
        codebook_size: int = 4096,
        num_codebooks: int = 10,
        semantic_begin_id: int = 151_678,
        semantic_end_id: int = 155_773,
        n_fast_layer: int = 4,
        fast_dim: int = 896,
        fast_n_head: int = 14,
        fast_n_local_heads: int = 2,
        fast_head_dim: int = 64,
        fast_intermediate_size: int = 4864,
        fast_attention_qkv_bias: bool = False,
        fast_attention_qk_norm: bool = False,
        fast_attention_o_bias: bool = False,
        norm_fastlayer_input: bool = True,
        codec_filename: str = "codec.pth",
        codec_sample_rate: int = SAMPLE_RATE_HZ,
        codec_frame_size: int = FRAME_SAMPLES,
        codec_post_n_layer: int = 8,
        codec_post_n_head: int = 16,
        codec_post_n_local_heads: int = 8,
        codec_post_intermediate_size: int = 1216,
        ras_window_size: int = 10,
        ras_temperature: float = 1.0,
        ras_top_p: float = 0.9,
        eos_token_id: int = 151_645,
        pad_token_id: int = 151_643,
        **checkpoint_fields: JsonValue,
    ) -> None:
        if max_seq_len <= 1 or semantic_begin_id < 0:
            raise ValueError(
                "Audio8 checkpoint has invalid prompt positions or semantic IDs"
            )
        elif (
            min(
                codec_post_n_layer,
                codec_post_n_head,
                codec_post_n_local_heads,
                codec_post_intermediate_size,
            )
            <= 0
        ):
            raise ValueError(
                "Audio8 checkpoint has invalid codec transformer dimensions"
            )
        else:
            pass
        # note (BeiningWu): Published configs contain both sets of backbone fields.
        checkpoint_fields.update(
            hidden_size=dim,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            num_key_value_heads=n_local_heads,
            max_position_embeddings=max_seq_len,
            rope_theta=rope_base,
            rms_norm_eps=norm_eps,
            attention_dropout=dropout,
        )
        super().__init__(
            vocab_size=vocab_size,
            intermediate_size=intermediate_size,
            tie_word_embeddings=tie_word_embeddings,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **checkpoint_fields,
        )
        self.dim: int = dim
        self.n_layer: int = n_layer
        self.n_head: int = n_head
        self.n_local_heads: int = n_local_heads
        self.head_dim: int = head_dim
        self.max_seq_len: int = max_seq_len
        self.rope_base: float = rope_base
        self.norm_eps: float = norm_eps
        self.dropout: float = dropout
        self.attention_qkv_bias: bool = attention_qkv_bias
        self.attention_qk_norm: bool = attention_qk_norm
        self.attention_o_bias: bool = attention_o_bias
        self.codebook_size: int = codebook_size
        self.num_codebooks: int = num_codebooks
        self.semantic_begin_id: int = semantic_begin_id
        self.semantic_end_id: int = semantic_end_id
        self.n_fast_layer: int = n_fast_layer
        self.fast_dim: int = fast_dim
        self.fast_n_head: int = fast_n_head
        self.fast_n_local_heads: int = fast_n_local_heads
        self.fast_head_dim: int = fast_head_dim
        self.fast_intermediate_size: int = fast_intermediate_size
        self.fast_attention_qkv_bias: bool = fast_attention_qkv_bias
        self.fast_attention_qk_norm: bool = fast_attention_qk_norm
        self.fast_attention_o_bias: bool = fast_attention_o_bias
        self.norm_fastlayer_input: bool = norm_fastlayer_input
        self.codec_filename: str = codec_filename
        self.codec_sample_rate: int = codec_sample_rate
        self.codec_frame_size: int = codec_frame_size
        self.codec_post_n_layer: int = codec_post_n_layer
        self.codec_post_n_head: int = codec_post_n_head
        self.codec_post_n_local_heads: int = codec_post_n_local_heads
        self.codec_post_intermediate_size: int = codec_post_intermediate_size
        self.ras_window_size: int = ras_window_size
        self.ras_temperature: float = ras_temperature
        self.ras_top_p: float = ras_top_p
