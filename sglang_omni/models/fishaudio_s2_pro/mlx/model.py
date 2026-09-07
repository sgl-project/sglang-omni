# SPDX-License-Identifier: Apache-2.0
"""Native MLX Fish Slow-AR and Fast-AR, using official checkpoint names.

The prompt and sampler live in Omni. This module owns only embeddings,
transformers, and their native caches; it has no mlx-audio dependency.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.cache import KVCache


class FishRoPE:
    def __init__(self, config):
        # Construct the canonical cached BF16 phases once on CPU. Recomputing
        # phases in another math library can round across a BF16 boundary.
        from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.utils import (
            precompute_freqs_cis,
        )

        phases = precompute_freqs_cis(
            config.max_seq_len, config.head_dim, config.rope_base
        )
        self.phases = mx.array(phases.float().numpy()).astype(mx.bfloat16)
        mx.eval(self.phases)

    def __call__(self, x, offset):
        end = offset + x.shape[2]
        if end > self.phases.shape[0]:
            raise ValueError("Fish MLX request exceeds the RoPE context")
        phases = self.phases[offset:end]
        even, odd = x[..., ::2].astype(mx.float32), x[..., 1::2].astype(mx.float32)
        cos, sin = phases[..., 0], phases[..., 1]
        return (
            mx.stack((even * cos - odd * sin, odd * cos + even * sin), axis=-1)
            .reshape(x.shape)
            .astype(x.dtype)
        )


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self._config = c
        self.wqkv = nn.Linear(
            c.dim,
            (c.n_head + 2 * c.n_local_heads) * c.head_dim,
            bias=c.attention_qkv_bias,
        )
        self.wo = nn.Linear(c.n_head * c.head_dim, c.dim, bias=c.attention_o_bias)
        if c.attention_qk_norm:
            self.q_norm = nn.RMSNorm(c.head_dim, eps=c.norm_eps)
            self.k_norm = nn.RMSNorm(c.head_dim, eps=c.norm_eps)

    def __call__(self, x, cache, rope):
        c = self._config
        b, s, _ = x.shape
        q, k, v = mx.split(
            self.wqkv(x),
            [c.n_head * c.head_dim, (c.n_head + c.n_local_heads) * c.head_dim],
            axis=-1,
        )
        q = q.reshape(b, s, c.n_head, c.head_dim).transpose(0, 2, 1, 3)
        k, v = (
            z.reshape(b, s, c.n_local_heads, c.head_dim).transpose(0, 2, 1, 3)
            for z in (k, v)
        )
        if c.attention_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = rope(q, cache.offset), rope(k, cache.offset)
        k, v = cache.update_and_fetch(k, v)
        y = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=c.head_dim**-0.5, mask="causal" if s > 1 else None
        )
        return self.wo(y.transpose(0, 2, 1, 3).reshape(b, s, -1))


class FeedForward(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w1 = nn.Linear(c.dim, c.intermediate_size, bias=False)
        self.w2 = nn.Linear(c.intermediate_size, c.dim, bias=False)
        self.w3 = nn.Linear(c.dim, c.intermediate_size, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attention = Attention(c)
        self.feed_forward = FeedForward(c)
        self.attention_norm = nn.RMSNorm(c.dim, eps=c.norm_eps)
        self.ffn_norm = nn.RMSNorm(c.dim, eps=c.norm_eps)

    def __call__(self, x, cache, rope):
        x = x + self.attention(self.attention_norm(x), cache, rope)
        return x + self.feed_forward(self.ffn_norm(x))


class Transformer(nn.Module):
    def __init__(self, c):
        super().__init__()
        if c.use_moe:
            raise ValueError("Fish MLX requires a dense checkpoint")
        self.embeddings = nn.Embedding(c.vocab_size, c.dim)
        self.layers = [TransformerBlock(c) for _ in range(c.n_layer)]
        self.norm = nn.RMSNorm(c.dim, eps=c.norm_eps)
        self._rope = FishRoPE(c)

    def make_cache(self, *, step=256):
        caches = [KVCache() for _ in self.layers]
        for cache in caches:
            cache.step = step
        return caches

    def __call__(self, x, cache):
        if len(cache) != len(self.layers):
            raise ValueError("Fish MLX cache layer count mismatch")
        for layer, state in zip(self.layers, cache):
            x = layer(x, state, self._rope)
        return self.norm(x)


class TextModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.model = Transformer(c)
        self._tied = c.tie_word_embeddings
        if not self._tied:
            self.lm_head = nn.Linear(c.dim, c.vocab_size, bias=False)

    def __call__(self, x, cache):
        hidden = self.model(x, cache)[:, -1]
        logits = (
            self.model.embeddings.as_linear(hidden)
            if self._tied
            else self.lm_head(hidden)
        )
        return logits, hidden


class AudioDecoder(Transformer):
    def __init__(self, c):
        super().__init__(c)
        self._config = c
        self.codebook_embeddings = nn.Embedding(
            c.vocab_size * c.num_codebooks, c.text_dim
        )
        self.project_in = (
            nn.Linear(c.text_dim, c.dim) if c.text_dim != c.dim else nn.Identity()
        )
        self.output = nn.Linear(c.dim, c.vocab_size, bias=False)

    def generate(self, hidden, semantic_code):
        # Fast-AR history lasts one frame. Evaluate the full greedy residual
        # chain together, without synchronizing the host for every codebook.
        cache = self.make_cache(step=self._config.num_codebooks + 1)
        self(self.project_in(hidden)[:, None], cache)
        code = mx.array([semantic_code], dtype=mx.int32)
        codes = [code]
        for _ in range(1, self._config.num_codebooks):
            hidden = self(self.embeddings(code)[:, None], cache)
            code = mx.argmax(self.output(hidden)[:, 0], axis=-1)
            codes.append(code)
        return mx.stack(codes, axis=-1)

    def mix_embeddings(self, text, codes):
        offsets = mx.arange(self._config.num_codebooks) * self._config.vocab_size
        vq = self.codebook_embeddings(codes + offsets).sum(axis=-2).astype(text.dtype)
        return (text + vq) * (self._config.num_codebooks + 1) ** -0.5


class FishModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = TextModel(SimpleNamespace(**config["text_config"]))
        self.audio_decoder = AudioDecoder(
            SimpleNamespace(**config["audio_decoder_config"])
        )
        self._config = config

    @classmethod
    def from_pretrained(cls, model_path):
        path = Path(model_path)
        config = json.loads((path / "config.json").read_text())
        if config.get("quantization") or config.get("quantization_config"):
            raise ValueError("Fish MLX currently requires unquantized BF16 weights")
        model = cls(config)
        weights = {}
        for shard in sorted(path.glob("*.safetensors")):
            loaded = mx.load(str(shard))
            if weights.keys() & loaded.keys():
                raise ValueError("Duplicate Fish MLX checkpoint tensor")
            weights.update(loaded)
        model.load_weights(list(weights.items()), strict=True)
        model.eval()
        mx.eval(model.parameters())
        return model


def to_mlx(tensor):
    tensor = tensor.detach().cpu()
    # NumPy has no BF16 dtype. Widening is exact and only used for reference
    # integer codes / test fixtures, not transformer activations in serving.
    if str(tensor.dtype) == "torch.bfloat16":
        return mx.array(tensor.float().numpy()).astype(mx.bfloat16)
    return mx.array(tensor.numpy())


def to_torch(array):
    import torch

    return torch.from_numpy(np.array(array.astype(mx.float32)))
