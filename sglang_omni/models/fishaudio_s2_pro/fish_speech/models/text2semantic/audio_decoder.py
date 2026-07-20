# SPDX-License-Identifier: Apache-2.0
"""Inference-only Fish fast audio decoder for S2-Pro serving."""

from __future__ import annotations

import math
import os
from typing import Callable

import torch
import torch.nn as nn
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from torch import Tensor
from torch.nn import functional as F
from transformers import PreTrainedModel

from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.configuration import (
    FishQwen3AudioDecoderConfig,
)
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.utils import (
    apply_rotary_emb,
    precompute_freqs_cis,
)

FISH_BATCH_INVARIANT = os.getenv("FISH_BATCH_INVARIANT", "false").lower() in (
    "true",
    "1",
    "yes",
)


@torch.library.custom_op(
    "mylib::flash_attn_kvcache", mutates_args=("k_cache", "v_cache")
)
def flash_attn_kvcache_op(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    cache_seqlens: torch.Tensor | None = None,
    causal: bool = False,
    num_splits: int = 0,
) -> torch.Tensor:
    return flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k,
        v=v,
        cache_seqlens=cache_seqlens.contiguous() if cache_seqlens is not None else None,
        causal=causal,
        num_splits=num_splits,
    )


@flash_attn_kvcache_op.register_fake
def _(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    cache_seqlens=None,
    causal=False,
    num_splits=0,
):
    return torch.empty_like(q)


class MyRMSNorm(nn.Module):
    """RMSNorm implementation used by Fish batch-invariant mode."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms = norm_x * (self.dim**-0.5)
        return x / (rms + self.eps) * self.weight


RMSNorm = MyRMSNorm if FISH_BATCH_INVARIANT else nn.RMSNorm


class Attention(nn.Module):
    """KV-cached attention used by the fast audio decoder."""

    def __init__(self, config: FishQwen3AudioDecoderConfig):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(
            config.dim, total_head_dim, bias=config.attention_qkv_bias
        )
        self.wo = nn.Linear(
            config.n_head * config.head_dim, config.dim, bias=config.attention_o_bias
        )

        if config.attention_qk_norm:
            self.q_norm = RMSNorm(config.head_dim, config.norm_eps)
            self.k_norm = RMSNorm(config.head_dim, config.norm_eps)

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.attention_qk_norm = config.attention_qk_norm
        self.kv_cache: KVCache | None = None

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        """Normalize legacy split-QKV checkpoints before strict loading."""
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward_kvcached(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        cache_seqlens: Tensor,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        q_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if self.attention_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        if self.kv_cache is None:
            raise RuntimeError("Fast audio decoder KV cache is not initialized")
        k_cache, v_cache = self.kv_cache.get(bsz)
        y = flash_attn_kvcache_op(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            causal=True,
            num_splits=1 if FISH_BATCH_INVARIANT else 0,
        )
        return self.wo(y.contiguous().view(bsz, seqlen, q_size))


class KVCache(nn.Module):
    """KV cache in the layout expected by ``flash_attn_with_kvcache``."""

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        cache_shape = (max_batch_size, max_seq_len, n_heads, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def get(self, batch_size: int) -> tuple[Tensor, Tensor]:
        return self.k_cache[:batch_size], self.v_cache[:batch_size]


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, dim: int, intermediate_size: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_size, bias=False)
        self.w3 = nn.Linear(dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x1, x3 = self.w1(x), self.w3(x)
        return self.w2(F.silu(x1) * x3)


class TransformerBlock(nn.Module):
    """Fast-decoder transformer block."""

    def __init__(self, config: FishQwen3AudioDecoderConfig) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(
            dim=config.dim, intermediate_size=config.intermediate_size
        )
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward_kvcached(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        cache_seqlens: Tensor,
    ) -> Tensor:
        h = x + self.attention.forward_kvcached(
            self.attention_norm(x),
            freqs_cis=freqs_cis,
            cache_seqlens=cache_seqlens,
        )
        return h + self.feed_forward(self.ffn_norm(h))


class FishQwen3AudioDecoder(PreTrainedModel):
    """Inference-only fast decoder for S2-Pro residual codebooks."""

    config_class = FishQwen3AudioDecoderConfig
    base_model_prefix = "audio_decoder"

    def __init__(self, config: FishQwen3AudioDecoderConfig):
        super().__init__(config)
        self.config = config

        if config.text_dim != config.dim:
            self.project_in = nn.Linear(config.text_dim, config.dim)
        else:
            self.project_in = nn.Identity()

        self.codebook_embeddings = nn.Embedding(
            config.vocab_size * config.num_codebooks,
            config.text_dim,
        )
        self.embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self._eager_forward_kvcached_layers: list[
            Callable[[Tensor, Tensor, Tensor], Tensor]
        ] = [layer.forward_kvcached for layer in self.layers]
        self._compiled_forward_kvcached_layers: (
            list[Callable[[Tensor, Tensor, Tensor], Tensor]] | None
        ) = None
        self._compiled_forward_kvcached_max_bs = 0
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.num_codebooks,
                config.head_dim,
                config.rope_base,
            ),
            persistent=False,
        )
        self.register_buffer(
            "codebook_offsets",
            torch.arange(config.num_codebooks) * config.vocab_size,
            persistent=False,
        )
        self.max_batch_size = -1

    def setup_caches(
        self,
        max_batch_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Allocate persistent KV/input-position buffers for decode."""
        if self.max_batch_size >= max_batch_size:
            return

        self.max_batch_size = max_batch_size
        device = next(self.parameters()).device
        max_seq_len = self.config.num_codebooks + 1
        for layer in self.layers:
            layer.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_len,
                self.config.n_local_heads,
                self.config.head_dim,
                dtype=dtype,
            ).to(device)

        self.register_buffer(
            "input_pos",
            torch.zeros(1, device=device, dtype=torch.long),
            persistent=False,
        )

    @property
    def kv_cache_max_batch_size(self) -> int:
        if not self.layers:
            raise RuntimeError("Audio decoder layers are not initialized")
        kv_cache = self.layers[0].attention.kv_cache
        if kv_cache is None:
            raise RuntimeError("Audio decoder KV cache is not initialized")
        return int(kv_cache.k_cache.shape[0])

    def reset_caches(self) -> None:
        """Reset all Fast AR KV caches."""
        for layer in self.layers:
            if layer.attention.kv_cache is not None:
                layer.attention.kv_cache.k_cache.zero_()
                layer.attention.kv_cache.v_cache.zero_()

    def set_compiled_forward_kvcached_layers(
        self,
        forward_kvcached_layers: list[Callable[[Tensor, Tensor, Tensor], Tensor]],
        *,
        max_batch_size: int,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if len(forward_kvcached_layers) != len(self.layers):
            raise ValueError("compiled layer count must match decoder layer count")
        self._compiled_forward_kvcached_layers = forward_kvcached_layers
        self._compiled_forward_kvcached_max_bs = max_batch_size

    def _select_forward_kvcached_layers(
        self, bsz: int
    ) -> list[Callable[[Tensor, Tensor, Tensor], Tensor]]:
        if (
            self._compiled_forward_kvcached_layers is not None
            and bsz <= self._compiled_forward_kvcached_max_bs
        ):
            return self._compiled_forward_kvcached_layers
        return self._eager_forward_kvcached_layers

    def forward_kvcached(self, x: Tensor, codebook_idx: int) -> Tensor:
        """Predict one residual codebook step with the persistent KV cache."""
        bsz = x.shape[0]
        self.input_pos.fill_(codebook_idx)
        freqs_cis = self.freqs_cis[self.input_pos]
        cache_seqlens = self.input_pos.expand(bsz).to(torch.int32)

        for layer in self._select_forward_kvcached_layers(bsz):
            x = layer(x, freqs_cis, cache_seqlens)

        return self.output(self.norm(x))

    def embed_text_dim(
        self,
        x: Tensor,
        vq_parts: Tensor | None = None,
        vq_mask_tokens: Tensor | None = None,
    ) -> Tensor:
        """Inject reference-code embeddings into Slow AR input embeddings."""
        if vq_parts is None or vq_mask_tokens is None:
            return x

        offset_parts = vq_parts + self.codebook_offsets[None, :]
        vq_embeds_sum = self.codebook_embeddings(offset_parts).sum(dim=1)
        vq_summed_embeds = x[vq_mask_tokens] + vq_embeds_sum.to(x.dtype)
        return vq_summed_embeds / math.sqrt(self.config.num_codebooks + 1)
