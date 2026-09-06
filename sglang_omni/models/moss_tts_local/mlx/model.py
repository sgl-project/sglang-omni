# SPDX-License-Identifier: Apache-2.0
"""Native MLX model for MOSS-TTS Local v1.5.

The module layout deliberately mirrors the official safetensors checkpoint.
Model structure was adapted from the MIT-licensed mlx-audio implementation.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
from mlx import nn
from mlx_lm.models.qwen3 import Qwen3Model

from .config import ModelConfig
from .local_transformer import LocalTransformer


class MossTTSLocalModel(nn.Module):
    def __init__(self, config: ModelConfig | dict) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        if config.model_type != "moss_tts_local":
            raise ValueError(
                f"expected model_type='moss_tts_local', got {config.model_type!r}"
            )
        self.config = config
        self.transformer = Qwen3Model(config.language_config)
        self.audio_embeddings = [
            nn.Embedding(config.audio_vocab_size, config.hidden_size)
            for _ in range(config.n_vq)
        ]
        self.text_lm_head = nn.Linear(
            config.hidden_size, config.language_config.vocab_size, bias=False
        )
        self.audio_lm_heads = [
            nn.Linear(config.hidden_size, config.audio_vocab_size, bias=False)
            for _ in range(config.n_vq)
        ]
        self.local_text_lm_head = nn.Linear(config.hidden_size, 2, bias=False)
        self.local_transformer = LocalTransformer(config.local_config())

    @property
    def layers(self) -> Sequence[nn.Module]:
        return self.transformer.layers

    def make_cache(self):
        from mlx_lm.models.cache import make_prompt_cache

        return make_prompt_cache(self.transformer)

    def input_embeddings(self, rows: mx.array) -> mx.array:
        if rows.ndim != 3 or rows.shape[-1] != self.config.channels:
            raise ValueError(
                f"expected rows shaped [batch, length, {self.config.channels}], "
                f"got {rows.shape}"
            )
        hidden = self.transformer.embed_tokens(rows[..., 0])
        for channel, embedding in enumerate(self.audio_embeddings):
            ids = rows[..., channel + 1]
            valid = ids != self.config.audio_pad_code
            safe_ids = mx.where(valid, ids, 0).astype(mx.int32)
            hidden = hidden + mx.where(valid[..., None], embedding(safe_ids), 0.0)
        return hidden

    def backbone(self, rows: mx.array, cache) -> mx.array:
        embeddings = self.input_embeddings(rows)
        dummy = mx.zeros(rows.shape[:2], dtype=mx.int32)
        return self.transformer(dummy, cache=cache, input_embeddings=embeddings)

    def decode_frame(
        self,
        global_hidden: mx.array,
        *,
        sample_text,
        sample_audio,
    ) -> mx.array:
        local_inputs = global_hidden[:, None, :]
        local_hidden = self.local_transformer(local_inputs)[:, -1, :]
        stop_choice = sample_text(self.local_text_lm_head(local_hidden))

        codes = []
        for channel in range(self.config.n_vq):
            code = sample_audio(self.audio_lm_heads[channel](local_hidden), channel)
            codes.append(code)
            if channel + 1 < self.config.n_vq:
                local_inputs = mx.concatenate(
                    [local_inputs, self.audio_embeddings[channel](code)[:, None, :]],
                    axis=1,
                )
                local_hidden = self.local_transformer(local_inputs)[:, -1, :]

        text = mx.where(
            stop_choice == 0,
            self.config.audio_assistant_slot_token_id,
            self.config.audio_end_token_id,
        ).astype(mx.int32)
        return mx.concatenate([text[:, None], mx.stack(codes, axis=-1)], axis=-1)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        return {
            key: value
            for key, value in weights.items()
            if not key.endswith(("rotary_emb.inv_freq", "position_ids"))
        }
