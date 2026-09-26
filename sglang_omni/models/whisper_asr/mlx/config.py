# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelConfig:
    """Whisper encoder-decoder config. Defaults match whisper-large-v3."""

    model_type: str = "whisper"

    num_mel_bins: int = 128
    d_model: int = 1280
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    max_source_positions: int = 1500

    decoder_layers: int = 32
    decoder_attention_heads: int = 20
    decoder_ffn_dim: int = 5120
    max_target_positions: int = 448
    vocab_size: int = 51866

    activation_function: str = "gelu"
    scale_embedding: bool = False

    decoder_start_token_id: int = 50258
    eos_token_id: int = 50257
    pad_token_id: int = 50256

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelConfig:
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
