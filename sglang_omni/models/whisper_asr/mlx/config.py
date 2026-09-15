# SPDX-License-Identifier: Apache-2.0
# Structure follows the Qwen3-ASR MLX path in
# sglang_omni/models/qwen3_asr/mlx/config.py.

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelConfig:
    """Configuration for the Whisper encoder-decoder model.

    Whisper's Hugging Face config is flat, so a single dataclass covers both
    the audio encoder and the text decoder. Defaults match
    ``openai/whisper-large-v3``.
    """

    model_type: str = "whisper"

    # Audio encoder
    num_mel_bins: int = 128
    d_model: int = 1280
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    max_source_positions: int = 1500

    # Text decoder
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
