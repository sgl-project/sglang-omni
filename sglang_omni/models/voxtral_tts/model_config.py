"""Parse Mistral-format params.json into structured config for Voxtral TTS."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VoxtralTextConfig:
    dim: int = 3072
    n_layers: int = 26
    head_dim: int = 128
    hidden_dim: int = 9216
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 131072
    rope_theta: float = 1000000.0
    norm_eps: float = 1e-5
    max_seq_len: int = 65536
    tied_embeddings: bool = True


@dataclass
class VoxtralAudioModelArgs:
    semantic_codebook_size: int = 8192
    acoustic_codebook_size: int = 21
    n_acoustic_codebook: int = 36
    acoustic_transformer_args: dict = field(default_factory=dict)
    audio_encoding_args: dict = field(default_factory=dict)
    audio_token_id: int = 24
    begin_audio_token_id: int = 25


@dataclass
class VoxtralModelConfig:
    text_config: VoxtralTextConfig = field(default_factory=VoxtralTextConfig)
    audio_model_args: VoxtralAudioModelArgs = field(
        default_factory=VoxtralAudioModelArgs
    )
    audio_tokenizer_args: dict = field(default_factory=dict)
    voices: dict[str, int] = field(default_factory=dict)
    model_path: str = ""

    @staticmethod
    def from_model_path(model_path: str) -> "VoxtralModelConfig":
        params_path = os.path.join(model_path, "params.json")
        with open(params_path) as f:
            params = json.load(f)

        text_config = VoxtralTextConfig(
            dim=params.get("dim", 3072),
            n_layers=params.get("n_layers", 26),
            head_dim=params.get("head_dim", 128),
            hidden_dim=params.get("hidden_dim", 9216),
            n_heads=params.get("n_heads", 32),
            n_kv_heads=params.get("n_kv_heads", 8),
            vocab_size=params.get("vocab_size", 131072),
            rope_theta=params.get("rope_theta", 1000000.0),
            norm_eps=params.get("norm_eps", 1e-5),
            max_seq_len=params.get("max_seq_len", 65536),
            tied_embeddings=params.get("tied_embeddings", True),
        )

        mm = params.get("multimodal", {})
        audio_args = mm.get("audio_model_args", {})
        audio_model_args = VoxtralAudioModelArgs(
            semantic_codebook_size=audio_args.get("semantic_codebook_size", 8192),
            acoustic_codebook_size=audio_args.get("acoustic_codebook_size", 21),
            n_acoustic_codebook=audio_args.get("n_acoustic_codebook", 36),
            acoustic_transformer_args=audio_args.get("acoustic_transformer_args", {}),
            audio_encoding_args=audio_args.get("audio_encoding_args", {}),
            audio_token_id=audio_args.get("audio_token_id", 24),
            begin_audio_token_id=audio_args.get("begin_audio_token_id", 25),
        )

        audio_tokenizer_args = mm.get("audio_tokenizer_args", {})
        voices = audio_tokenizer_args.pop("voice", {})

        return VoxtralModelConfig(
            text_config=text_config,
            audio_model_args=audio_model_args,
            audio_tokenizer_args=audio_tokenizer_args,
            voices=voices,
            model_path=model_path,
        )

    def to_hf_mistral_config(self) -> Any:
        """Build a HuggingFace MistralConfig for loading transformers.MistralModel."""
        from transformers import MistralConfig

        return MistralConfig(
            hidden_size=self.text_config.dim,
            intermediate_size=self.text_config.hidden_dim,
            num_hidden_layers=self.text_config.n_layers,
            num_attention_heads=self.text_config.n_heads,
            num_key_value_heads=self.text_config.n_kv_heads,
            head_dim=self.text_config.head_dim,
            vocab_size=self.text_config.vocab_size,
            max_position_embeddings=self.text_config.max_seq_len,
            rope_theta=self.text_config.rope_theta,
            rms_norm_eps=self.text_config.norm_eps,
            tie_word_embeddings=self.text_config.tied_embeddings,
            sliding_window=None,
        )
