# SPDX-License-Identifier: Apache-2.0
"""Hidden-state BailingMoE decoder matching Ming's TTS path."""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache
from mlx_lm.models.switch_layers import SwitchGLU

from .config import TextConfig


class MRotaryEmbedding(nn.Module):
    def __init__(self, config: TextConfig) -> None:
        super().__init__()
        self._sections = config.mrope_section
        self._inv_freq = 1.0 / (
            config.rope_theta
            ** (mx.arange(0, config.head_dim, 2, dtype=mx.float32) / config.head_dim)
        )
        # Private arrays are excluded from parameters(); finish before thread handoff.
        mx.eval(self._inv_freq)

    def __call__(self, x: mx.array, positions: mx.array) -> mx.array:
        # positions: [3, batch, sequence]; x: [batch, heads, sequence, dim].
        phases = positions[..., None].astype(mx.float32) * self._inv_freq
        parts = []
        start = 0
        for axis, width in enumerate(self._sections):
            parts.append(phases[axis, ..., start:start + width])
            start += width
        phase = mx.concatenate(parts, axis=-1)[:, None]
        left, right = mx.split(x.astype(mx.float32), 2, axis=-1)
        cos, sin = mx.cos(phase), mx.sin(phase)
        return mx.concatenate(
            (left * cos - right * sin, right * cos + left * sin), axis=-1
        ).astype(x.dtype)


class BailingMoeAttention(nn.Module):
    def __init__(self, config: TextConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.query_key_value = nn.Linear(
            config.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=config.use_qkv_bias,
        )
        self.dense = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=config.use_bias
        )
        self.rotary_emb = MRotaryEmbedding(config)

    def __call__(
        self,
        x: mx.array,
        positions: mx.array,
        mask: mx.array | str | None,
        cache: KVCache | None,
    ) -> mx.array:
        batch, length, _ = x.shape
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = mx.split(
            self.query_key_value(x), (q_size, q_size + kv_size), axis=-1
        )
        q = q.reshape(batch, length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, length, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, length, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k = self.rotary_emb(q, positions), self.rotary_emb(k, positions)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.head_dim**-0.5, mask=mask
        )
        return self.dense(out.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class BailingMoeMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class BailingMoeSparseMoeBlock(nn.Module):
    def __init__(self, config: TextConfig) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        if config.multi_gate:
            # Loaded for coverage; Ming TTS does not pass modality routing masks.
            self.image_gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
            self.audio_gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = SwitchGLU(
            config.hidden_size, config.moe_intermediate_size, config.num_experts
        )
        self.shared_experts = (
            BailingMoeMLP(
                config.hidden_size,
                config.moe_intermediate_size * config.num_shared_experts,
            )
            if config.num_shared_experts else None
        )

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        # Keep the router linear in weight dtype, then softmax in FP32 like Omni.
        logits = self.gate(x.astype(self.gate.weight.dtype)).astype(x.dtype)
        scores = mx.softmax(logits.astype(mx.float32), axis=-1)
        indices = mx.argpartition(-scores, kth=self.top_k - 1, axis=-1)[
            ..., :self.top_k
        ]
        weights = mx.take_along_axis(scores, indices, axis=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        return indices, weights * self.routed_scaling_factor

    def __call__(self, x: mx.array) -> mx.array:
        indices, weights = self.route(x)
        routed = self.experts(x, indices)
        out = (routed.astype(mx.float32) * weights[..., None]).sum(axis=-2)
        out = out.astype(x.dtype)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out


class BailingMoeDecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_id: int) -> None:
        super().__init__()
        self.attention = BailingMoeAttention(config)
        self.mlp = (
            BailingMoeSparseMoeBlock(config)
            if layer_id >= config.first_k_dense_replace
            else BailingMoeMLP(config.hidden_size, config.intermediate_size)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        positions: mx.array,
        mask: mx.array | str | None,
        cache: KVCache | None,
    ) -> mx.array:
        x = x + self.attention(self.input_layernorm(x), positions, mask, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class BailingMoeTextModel(nn.Module):
    def __init__(self, config: TextConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            BailingMoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array | None = None,
        *,
        inputs_embeds: mx.array | None = None,
        positions: mx.array | None = None,
        cache: Sequence[KVCache | None] | None = None,
    ) -> mx.array:
        x = self.word_embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)
        if positions is None:
            offset = cache[0].offset if cache[0] is not None else 0
            positions = mx.broadcast_to(
                mx.arange(offset, offset + x.shape[1]),
                (3, x.shape[0], x.shape[1]),
            )
        elif positions.shape != (3, x.shape[0], x.shape[1]):
            raise ValueError("positions must have shape [3, batch, sequence]")
        mask = create_attention_mask(x, cache[0])
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            x = layer(x, positions, mask, layer_cache)
        return self.norm(x)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]
