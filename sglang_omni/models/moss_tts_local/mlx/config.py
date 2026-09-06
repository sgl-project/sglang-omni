# SPDX-License-Identifier: Apache-2.0
"""MLX configuration adapters for the MOSS-TTS Local checkpoint."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from mlx_lm.models.qwen3 import ModelArgs as Qwen3ModelArgs


@dataclass
class GPT2Config:
    vocab_size: int = 0
    n_positions: int = 13
    n_ctx: int = 13
    n_embd: int = 2560
    n_layer: int = 1
    n_head: int = 32
    n_inner: int | None = None
    activation_function: str = "silu"
    layer_norm_epsilon: float = 1e-6
    scale_attn_weights: bool = True
    scale_attn_by_inverse_layer_idx: bool = False
    position_embedding_type: str = "rope"
    rope_base: float = 1_000_000.0

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> GPT2Config:
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in fields})


@dataclass
class ModelConfig:
    model_type: str
    language_config: Qwen3ModelArgs
    gpt2_config: GPT2Config
    n_vq: int = 12
    audio_vocab_size: int = 1024
    local_transformer_layers: int = 1
    local_text_head_mode: str = "binary"
    audio_assistant_slot_token_id: int = 151656
    audio_end_token_id: int = 151670
    audio_pad_code: int = 1024

    @property
    def hidden_size(self) -> int:
        return int(self.language_config.hidden_size)

    @property
    def channels(self) -> int:
        return self.n_vq + 1

    def local_config(self) -> GPT2Config:
        return replace(
            self.gpt2_config,
            n_positions=self.channels,
            n_ctx=self.channels,
            n_layer=self.local_transformer_layers,
        )

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ModelConfig:
        language_values = dict(
            values.get("language_config") or values.get("qwen3_config") or {}
        )
        language_values.setdefault("model_type", "qwen3")
        language_values.setdefault("tie_word_embeddings", True)
        return cls(
            model_type=str(values.get("model_type", "moss_tts_local")),
            language_config=Qwen3ModelArgs.from_dict(language_values),
            gpt2_config=GPT2Config.from_dict(dict(values.get("gpt2_config") or {})),
            n_vq=int(values.get("n_vq", 12)),
            audio_vocab_size=int(values.get("audio_vocab_size", 1024)),
            local_transformer_layers=int(values.get("local_transformer_layers", 1)),
            local_text_head_mode=str(values.get("local_text_head_mode", "binary")),
            audio_assistant_slot_token_id=int(
                values.get("audio_assistant_slot_token_id", 151656)
            ),
            audio_end_token_id=int(values.get("audio_end_token_id", 151670)),
            audio_pad_code=int(values.get("audio_pad_code", 1024)),
        )
