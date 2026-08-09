# SPDX-License-Identifier: Apache-2.0
"""Hugging Face configuration for Kimi-Audio checkpoints."""

from __future__ import annotations

from transformers import AutoConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


class KimiAudioConfig(Qwen2Config):
    model_type = "moonshot_kimia"

    def __init__(
        self,
        vocab_size: int = 168448,
        hidden_size: int = 3584,
        intermediate_size: int = 18944,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 28,
        num_key_value_heads: int | None = 4,
        kimia_mimo_layers: int = 6,
        kimia_mimo_audiodelaytokens: int = 5,
        kimia_mimo_transformer_from_layer_index: int = 21,
        kimia_audio_output_vocab: int = 16896,
        kimia_text_output_vocab: int = 152064,
        num_audio_special_tokens: int = 512,
        num_base_tokens: int = 151643,
        kimia_token_offset: int = 152064,
        use_whisper_feature: bool = True,
        kimia_adaptor_input_dim: int = 5120,
        kimia_media_begin: int = 151661,
        kimia_media_end: int = 151663,
        kimia_text_blank: int = 151666,
        kimia_text_eos: int = 151667,
        **kwargs,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            **kwargs,
        )
        self.kimia_mimo_layers = kimia_mimo_layers
        self.kimia_mimo_audiodelaytokens = kimia_mimo_audiodelaytokens
        self.kimia_mimo_transformer_from_layer_index = (
            kimia_mimo_transformer_from_layer_index
        )
        self.kimia_audio_output_vocab = kimia_audio_output_vocab
        self.kimia_text_output_vocab = kimia_text_output_vocab
        self.num_audio_special_tokens = num_audio_special_tokens
        self.num_base_tokens = num_base_tokens
        self.kimia_token_offset = kimia_token_offset
        self.use_whisper_feature = use_whisper_feature
        self.kimia_adaptor_input_dim = kimia_adaptor_input_dim
        self.kimia_media_begin = kimia_media_begin
        self.kimia_media_end = kimia_media_end
        self.kimia_text_blank = kimia_text_blank
        self.kimia_text_eos = kimia_text_eos


AutoConfig.register("moonshot_kimia", KimiAudioConfig, exist_ok=True)

__all__ = ["KimiAudioConfig"]
