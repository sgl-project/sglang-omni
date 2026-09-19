# SPDX-License-Identifier: Apache-2.0
"""MLX config for the Chatterbox-Turbo T3 backbone."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChatterboxT3MlxConfig:
    model_type: str = "chatterbox_t3"
    n_ctx: int = 8196
    n_embd: int = 1024
    n_head: int = 16
    n_layer: int = 24
    n_positions: int = 8196
    layer_norm_epsilon: float = 1e-5
    text_vocab_size: int = 50276
    speech_vocab_size: int = 6563
    speaker_embed_size: int = 256
