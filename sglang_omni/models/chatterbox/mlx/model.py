# SPDX-License-Identifier: Apache-2.0
"""Native MLX T3 model for Chatterbox-Turbo (GPT-2 backbone + dual vocab head)."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.gpt2 import ModelArgs as GPT2ModelArgs

from sglang_omni.models.chatterbox.mlx.config import ChatterboxT3MlxConfig

_START_SPEECH_TOKEN = 6561


class NoPE(nn.Module):
    """Identity standing in for RoPE.

    GPT-2 encodes positions additively via wpe at the input, so its KV
    already carries positional info and the SGLang batched-decode wrapper's
    unconditional rope(x, offset=...) call must be a no-op. Mirrors the
    upstream NoPE in SGLang's MLX backend.
    """

    dims = 0
    traditional = True

    def __call__(self, x: mx.array, offset: Any = 0) -> mx.array:
        return x


class ChatterboxT3Attention(nn.Module):
    """GPT-2 multi-head attention reshaped to the SGLang MLX contract.

    The checkpoint stores a fused c_attn Conv1D; the loader splits it into
    q_proj/k_proj/v_proj so the SGLang batched-decode wrapper can drive the
    projections independently.
    """

    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        self.num_attention_heads = n_head
        self.num_key_value_heads = n_head
        self.head_dim = n_embd // n_head
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.o_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.rope = NoPE()

    def __call__(
        self,
        x: mx.array,
        mask: Any = None,
        cache: Any = None,
    ) -> mx.array:
        B, L, D = x.shape
        H, Hk, Hd = self.num_attention_heads, self.num_key_value_heads, self.head_dim

        queries = self.q_proj(x).reshape(B, L, H, Hd).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, Hk, Hd).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, Hk, Hd).transpose(0, 2, 1, 3)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class ChatterboxT3MLP(nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(nn.gelu_approx(self.c_fc(x)))


class ChatterboxT3Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, eps: float) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, eps=eps)
        self.attn = ChatterboxT3Attention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd, eps=eps)
        self.mlp = ChatterboxT3MLP(n_embd)

    def __call__(
        self,
        x: mx.array,
        mask: Any = None,
        cache: Any = None,
    ) -> mx.array:
        hidden = x + self.attn(self.ln_1(x), mask, cache)
        return hidden + self.mlp(self.ln_2(hidden))


class ChatterboxT3MlxModel(nn.Module):
    """T3 AR model: 24 GPT-2 blocks, dual text/speech embedding and head, plus a
    speaker conditioning prefix projected by cond_enc."""

    def __init__(self, config: ChatterboxT3MlxConfig) -> None:
        super().__init__()
        self.config = config
        self.h = [
            ChatterboxT3Block(config.n_embd, config.n_head, config.layer_norm_epsilon)
            for _ in range(config.n_layer)
        ]
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.text_emb = nn.Embedding(config.text_vocab_size, config.n_embd)
        self.speech_emb = nn.Embedding(config.speech_vocab_size, config.n_embd)
        self.text_head = nn.Linear(config.n_embd, config.text_vocab_size, bias=False)
        self.speech_head = nn.Linear(config.n_embd, config.speech_vocab_size, bias=True)
        self.cond_enc = nn.Linear(config.speaker_embed_size, config.n_embd, bias=True)

    def _build_inputs_embeds(
        self,
        speaker_emb: mx.array,
        cond_speech_tokens: mx.array,
        text_tokens: mx.array,
    ) -> mx.array:
        # Sequence order matches T3.prepare_input_embeds:
        # [cond_enc(speaker), speech_emb(cond speech), text_emb(text), speech_emb(start)]
        cond_spkr = self.cond_enc(speaker_emb)[:, None]
        cond_speech = self.speech_emb(cond_speech_tokens)
        text = self.text_emb(text_tokens)
        start = self.speech_emb(
            mx.full((text_tokens.shape[0], 1), _START_SPEECH_TOKEN, dtype=mx.int32)
        )
        return mx.concatenate([cond_spkr, cond_speech, text, start], axis=1)

    def _forward_last_logits(self, inputs_embeds: mx.array, cache: Any = None) -> mx.array:
        _, length, _ = inputs_embeds.shape
        if cache is None:
            cache = [None] * len(self.h)
        offset = cache[0].offset if cache[0] is not None else 0
        position_ids = mx.arange(length) + offset
        hidden = inputs_embeds + self.wpe(position_ids)
        mask = create_attention_mask(hidden, cache[0])
        for layer, layer_cache in zip(self.h, cache):
            hidden = layer(hidden, mask, cache=layer_cache)
        hidden = self.ln_f(hidden)
        return self.speech_head(hidden)

    def __call__(self, inputs: mx.array, cache: Any = None) -> mx.array:
        return self._forward_last_logits(self.speech_emb(inputs), cache=cache)

    @property
    def layers(self) -> list[nn.Module]:
        return self.h
