# SPDX-License-Identifier: Apache-2.0
"""Frame embedding, codebook heads, and dense depth decoder for CSM-1B.

Port of HuggingFace ``transformers.models.csm``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang_omni.models.csm_tts import sampler

# Hopper cuDNN SDPA aborts on later T=1 depth steps (additive causal mask +
# GQA). Guard the full 31-step frame, not a single T=2 probe.
_DEPTH_SDPA_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Half-rotate for RoPE. Checkpoint Q/K already use this permutation."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def llama3_scaled_inv_freq(
    rope_theta: float, dim: int, rope_scaling: dict[str, Any]
) -> torch.Tensor:
    """Llama-3 scaled RoPE inverse frequencies (attention_factor is 1.0)."""
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, dim, 2, dtype=torch.int64).to(torch.float32) / dim)
    )
    factor = float(rope_scaling["factor"])
    low_freq_factor = float(rope_scaling["low_freq_factor"])
    high_freq_factor = float(rope_scaling["high_freq_factor"])
    old_context_len = float(rope_scaling["original_max_position_embeddings"])

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(
        wavelen > low_freq_wavelen, inv_freq / factor, inv_freq
    )
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (
        1 - smooth_factor
    ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)


class CsmRMSNorm(nn.Module):
    """RMSNorm with fp32 math, then multiply by weight in the input dtype."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class CsmFrameEmbedding(nn.Module):
    """Fused 32-codebook table ``[65632, 2048]``, tied to the depth decoder."""

    def __init__(
        self,
        num_codebooks: int = 32,
        codebook_vocab: int = 2051,
        hidden_size: int = 2048,
    ) -> None:
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_vocab = codebook_vocab
        self.weight = nn.Parameter(
            torch.empty(num_codebooks * codebook_vocab, hidden_size)
        )
        self.register_buffer(
            "audio_tokens_offsets",
            torch.arange(num_codebooks, dtype=torch.int64) * codebook_vocab,
            persistent=False,
        )

    def forward(self, codes_B32: torch.Tensor) -> torch.Tensor:
        """Sum of per-codebook lookups for one frame (``[B, 32]`` → ``[B, hidden]``)."""
        return F.embedding(codes_B32 + self.audio_tokens_offsets, self.weight).sum(
            dim=-2
        )

    def embed_codebook(self, codes_B: torch.Tensor, k: int) -> torch.Tensor:
        """Lookup codebook ``k`` (depth position ``p`` uses offset ``(p - 1)``)."""
        return F.embedding(codes_B + k * self.codebook_vocab, self.weight)


class CsmCodebooksHead(nn.Module):
    """Per-depth-step heads. weight stays [31, 1024, 2051] as in the checkpoint."""

    def __init__(
        self,
        num_heads: int = 31,
        hidden_size: int = 1024,
        codebook_vocab: int = 2051,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, hidden_size, codebook_vocab))

    def forward(self, h_BD: torch.Tensor, step: int) -> torch.Tensor:
        """Logits for codebook ``step + 1``. Caller casts to fp32 before sampling."""
        return h_BD @ self.weight[step]


class CsmAttention(nn.Module):
    """SDPA + Llama-3 RoPE + GQA. Attention only reads the written KV prefix."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.num_heads = int(cfg.num_attention_heads)
        self.num_kv_heads = int(cfg.num_key_value_heads)
        self.head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // self.num_heads))
        hidden = int(cfg.hidden_size)
        bias = bool(getattr(cfg, "attention_bias", False))
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=bias)

    def forward(
        self,
        hidden_BTD: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        write_pos: int,
    ) -> torch.Tensor:
        B, T, _ = hidden_BTD.shape
        q = self.q_proj(hidden_BTD).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_BTD).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_BTD).view(B, T, self.num_kv_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos = rope_cos[write_pos : write_pos + T].to(q.dtype)
        sin = rope_sin[write_pos : write_pos + T].to(q.dtype)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        k_cache[:, :, write_pos : write_pos + T] = k
        v_cache[:, :, write_pos : write_pos + T] = v
        end = write_pos + T
        keys = k_cache[:, :, :end]
        vals = v_cache[:, :, :end]
        rep = self.num_heads // self.num_kv_heads
        keys = keys.repeat_interleave(rep, dim=1)
        vals = vals.repeat_interleave(rep, dim=1)
        # Prefill is causal over the new span; decode attends the written prefix.
        if write_pos == 0 and T > 1:
            out = F.scaled_dot_product_attention(q, keys, vals, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, keys, vals, is_causal=False)
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)


class CsmMLP(nn.Module):
    """SwiGLU MLP. Submodule names match the checkpoint."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        hidden = int(cfg.hidden_size)
        inter = int(cfg.intermediate_size)
        bias = bool(getattr(cfg, "mlp_bias", False))
        self.gate_proj = nn.Linear(hidden, inter, bias=bias)
        self.up_proj = nn.Linear(hidden, inter, bias=bias)
        self.down_proj = nn.Linear(inter, hidden, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CsmDecoderLayer(nn.Module):
    """RMSNorm → SDPA → RMSNorm → SwiGLU for the depth decoder."""

    def __init__(self, cfg: Any, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = CsmAttention(cfg)
        self.mlp = CsmMLP(cfg)
        self.input_layernorm = CsmRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = CsmRMSNorm(
            cfg.hidden_size, eps=cfg.rms_norm_eps
        )

    def forward(
        self,
        hidden_BTD: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        write_pos: int,
    ) -> torch.Tensor:
        residual = hidden_BTD
        hidden = self.input_layernorm(hidden_BTD)
        hidden = self.self_attn(hidden, rope_cos, rope_sin, k_cache, v_cache, write_pos)
        hidden = residual + hidden
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        return residual + hidden


class CsmDepthDecoder(nn.Module):
    """31-step inner AR per 80 ms frame. Depth KV is slot-indexed and frame-local."""

    def __init__(
        self,
        depth_cfg: Any,
        frame_embedding: CsmFrameEmbedding,
        max_slots: int,
    ) -> None:
        """``frame_embedding`` is the tied table; ``max_slots`` sizes the static KV."""
        super().__init__()
        self.config = depth_cfg
        self.max_slots = max_slots
        self.num_codebooks = depth_cfg.num_codebooks
        self.frame_embedding = frame_embedding
        self.inputs_embeds_projector = nn.Linear(
            depth_cfg.backbone_hidden_size, depth_cfg.hidden_size, bias=False
        )
        self.layers = nn.ModuleList(
            CsmDecoderLayer(depth_cfg, i) for i in range(depth_cfg.num_hidden_layers)
        )
        self.norm = CsmRMSNorm(depth_cfg.hidden_size, eps=depth_cfg.rms_norm_eps)
        self.codebooks_head = CsmCodebooksHead(
            depth_cfg.num_codebooks - 1, depth_cfg.hidden_size, depth_cfg.vocab_size
        )

        ctx = depth_cfg.max_position_embeddings
        inv_freq = llama3_scaled_inv_freq(
            float(depth_cfg.rope_theta), depth_cfg.head_dim, depth_cfg.rope_scaling
        )
        freqs = torch.outer(torch.arange(ctx, dtype=torch.float32), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("rope_cos", emb.cos(), persistent=False)
        self.register_buffer("rope_sin", emb.sin(), persistent=False)

        kv_shape = (
            depth_cfg.num_hidden_layers,
            max_slots,
            depth_cfg.num_key_value_heads,
            ctx,
            depth_cfg.head_dim,
        )
        self.register_buffer("k_cache", torch.zeros(kv_shape), persistent=False)
        self.register_buffer("v_cache", torch.zeros(kv_shape), persistent=False)

    def forward_step(
        self,
        embeds_BT2048: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        write_pos: int,
    ) -> torch.Tensor:
        """Project and run all layers from write_pos; return last-token hidden."""
        hidden = self.inputs_embeds_projector(embeds_BT2048)
        for i, layer in enumerate(self.layers):
            hidden = layer(
                hidden,
                self.rope_cos,
                self.rope_sin,
                k_cache[i],
                v_cache[i],
                write_pos,
            )
        return self.norm(hidden[:, -1])

    def generate_frame(
        self,
        h_B2048: torch.Tensor,
        cb0_B: torch.Tensor,
        temps_B: torch.Tensor,
        top_ks_B: torch.Tensor,
        *,
        bs: int,
    ) -> torch.Tensor:
        """Sample cb1..cb31 for one frame. cb31 is never forwarded."""
        with torch.no_grad(), sdpa_kernel(_DEPTH_SDPA_BACKENDS):
            return self.generate_frame_unguarded(
                h_B2048, cb0_B, temps_B, top_ks_B, bs=bs
            )

    def generate_frame_unguarded(
        self,
        h_B2048: torch.Tensor,
        cb0_B: torch.Tensor,
        temps_B: torch.Tensor,
        top_ks_B: torch.Tensor,
        *,
        bs: int,
    ) -> torch.Tensor:
        assert bs <= self.max_slots, f"bs={bs} exceeds depth KV slots {self.max_slots}"
        assert h_B2048.shape[0] == bs and cb0_B.shape[0] == bs
        temps = temps_B[:bs]
        top_ks = top_ks_B[:bs]
        k_cache = self.k_cache[:, :bs]
        v_cache = self.v_cache[:, :bs]

        embeds = torch.stack(
            [h_B2048, self.frame_embedding.embed_codebook(cb0_B, 0)], dim=1
        )
        h_last = self.forward_step(embeds, k_cache, v_cache, write_pos=0)
        prev = sampler.sample_codes_batched(
            self.codebooks_head(h_last, 0).float(), temps, top_ks
        )
        codes = [cb0_B, prev]

        # Position p embeds cb_{p-1}; the last sampled codebook is never forwarded.
        for p in range(2, self.num_codebooks):
            embeds = self.frame_embedding.embed_codebook(prev, p - 1).unsqueeze(1)
            h_last = self.forward_step(embeds, k_cache, v_cache, write_pos=p)
            prev = sampler.sample_codes_batched(
                self.codebooks_head(h_last, p - 1).float(), temps, top_ks
            )
            codes.append(prev)

        return torch.stack(codes, dim=1)


__all__ = [
    "CsmAttention",
    "CsmCodebooksHead",
    "CsmDecoderLayer",
    "CsmDepthDecoder",
    "CsmFrameEmbedding",
    "CsmMLP",
    "CsmRMSNorm",
]
