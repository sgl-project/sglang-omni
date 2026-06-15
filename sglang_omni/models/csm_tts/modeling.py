# SPDX-License-Identifier: Apache-2.0
# CSM-1B architecture (frame embedding, codebooks head, dense depth decoder) ports the
# reference implementation in HuggingFace Transformers `transformers/models/csm` (Apache-2.0).
"""Framework-free torch modules for CSM-1B: frame embedding, codebooks head,
dense depth decoder (4L, static 33-position KV, 31-step inner AR).

These compose with an SGLang ``LlamaForCausalLM`` backbone in
:class:`sglang_omni.models.csm_tts.model.CsmTTSModel`. Everything here is
plain eager torch with static shapes, CUDA-graph-friendly (capture is not
yet wired; eager by default): the 31-iteration host loop unrolls at capture
with no data-dependent branches.


"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from sglang_omni.models.csm_tts import sampler


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HF half-rotation rope application (modeling_csm.py:201-205). The HF
    shards already carry the interleaved→half-rotated Q/K permutation
    (convert_csm.py:61-67), so this is the matching apply."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _llama3_scaled_inv_freq(
    rope_theta: float, dim: int, rope_scaling: dict[str, Any]
) -> torch.Tensor:
    """llama3-scaled RoPE inverse frequencies, exact port of HF
    ``modeling_rope_utils.py`` ``_compute_llama3_parameters`` (the
    attention_factor for this rope type is 1.0, so cos/sin stay unscaled).
    """
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
    """HF-exact RMSNorm (modeling_csm.py:99-114): fp32 internal math, cast
    back to the input dtype before the weight multiply."""

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
    """Fused 32-codebook audio embedding: ONE ``weight [65632, 2048]`` table
    (32 * 2051 rows; tied to the depth decoder's embed table).

    Mirrors HF ``modeling_csm.py:655-666``: register
    ``audio_tokens_offsets = arange(32) * 2051`` as a buffer.
    """

    def __init__(
        self,
        num_codebooks: int = 32,
        codebook_vocab: int = 2051,
        hidden_size: int = 2048,
    ) -> None:
        """Owns ``weight: nn.Parameter [num_codebooks * codebook_vocab, hidden_size]``
        (= ``[65632, 2048]``, bf16 after model cast) and the int64
        ``audio_tokens_offsets [32]`` buffer."""
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
        """Frame embed: sum of per-codebook lookups.

        ``F.embedding(codes_B32 + audio_tokens_offsets).sum(-2)``.

        Args:
            codes_B32: int64 ``[B, 32]``, values in ``[0, 2050]``.

        Returns:
            ``[B, 2048]`` (weight dtype, bf16 in the ship config).
        """
        return F.embedding(codes_B32 + self.audio_tokens_offsets, self.weight).sum(
            dim=-2
        )

    def embed_codebook(self, codes_B: torch.Tensor, k: int) -> torch.Tensor:
        """Single-codebook lookup at table offset ``k * 2051`` (depth-step
        feedback: depth position ``p`` embeds the previous code with offset
        ``(p - 1) * 2051``).

        Args:
            codes_B: int64 ``[B]``, values in ``[0, 2050]``.
            k: codebook index in ``[0, 31]``.

        Returns:
            ``[B, 2048]`` (weight dtype).
        """
        return F.embedding(codes_B + k * self.codebook_vocab, self.weight)


class CsmCodebooksHead(nn.Module):
    """Per-depth-step output heads, ``weight [31, 1024, 2051]`` stored EXACTLY
    in checkpoint layout. Head *k* predicts codebook *k+1*.

    ⚠ Orientation trap: the forward is
    ``h @ weight[step]`` (≡ ``F.linear(h, weight[step].T)``, HF
    ``modeling_csm.py:528-536``). Do NOT "fix" shapes by re-storing the weight
    as ``[31, 2051, 1024]`` — combined with loading the checkpoint tensor
    ``(31, 1024, 2051)`` untransposed, that RUNS and produces garbage audio.
    The tiny-config depth parity test pins this orientation and is a hard
    correctness blocker.
    """

    def __init__(
        self,
        num_heads: int = 31,
        hidden_size: int = 1024,
        codebook_vocab: int = 2051,
    ) -> None:
        """Owns ``weight: nn.Parameter [31, 1024, 2051]`` (checkpoint layout)."""
        super().__init__()
        self.num_heads = num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, hidden_size, codebook_vocab))

    def forward(self, h_BD: torch.Tensor, step: int) -> torch.Tensor:
        """Logits for depth step ``step`` (predicting codebook ``step + 1``).

        Args:
            h_BD: ``[B, 1024]`` depth hidden (bf16).
            step: head index in ``[0, 30]``.

        Returns:
            ``[B, 2051]`` logits in the WEIGHT dtype — caller casts to fp32
            before sampling.
        """
        return h_BD @ self.weight[step]


class _CsmDepthAttention(nn.Module):
    """SDPA attention over the fixed 33-length static KV. Submodule names
    (``{q,k,v,o}_proj``) match the checkpoint layer keys so the manual
    weight copies stay 1:1."""

    def __init__(self, depth_cfg: Any) -> None:
        super().__init__()
        self.num_heads = depth_cfg.num_attention_heads
        self.num_kv_heads = depth_cfg.num_key_value_heads
        self.head_dim = depth_cfg.head_dim
        hidden = depth_cfg.hidden_size
        bias = depth_cfg.attention_bias
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
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = hidden_BTD.shape
        q = self.q_proj(hidden_BTD).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_BTD).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_BTD).view(B, T, self.num_kv_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # HF casts cos/sin to the activation dtype before applying
        # (modeling_csm.py:182, 209-231) — [T, head_dim] broadcasts over heads.
        cos = rope_cos[write_pos : write_pos + T].to(q.dtype)
        sin = rope_sin[write_pos : write_pos + T].to(q.dtype)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        k_cache[:, :, write_pos : write_pos + T] = k
        v_cache[:, :, write_pos : write_pos + T] = v

        # GQA: kv head g serves q heads [g*rep, (g+1)*rep) — same ordering as
        # HF repeat_kv (modeling_csm.py:234-243).
        rep = self.num_heads // self.num_kv_heads
        keys = k_cache.repeat_interleave(rep, dim=1)
        vals = v_cache.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, keys, vals, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)


class _CsmDepthMLP(nn.Module):
    """SwiGLU MLP (1024 → 8192 → 1024); submodule names match checkpoint keys."""

    def __init__(self, depth_cfg: Any) -> None:
        super().__init__()
        hidden = depth_cfg.hidden_size
        inter = depth_cfg.intermediate_size
        bias = depth_cfg.mlp_bias
        self.gate_proj = nn.Linear(hidden, inter, bias=bias)
        self.up_proj = nn.Linear(hidden, inter, bias=bias)
        self.down_proj = nn.Linear(inter, hidden, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CsmDepthDecoderLayer(nn.Module):
    """One depth-decoder layer: RMSNorm → SDPA attention (8 heads, head_dim
    128, 2 KV heads GQA, llama3-scaled rope) → RMSNorm → SwiGLU
    (1024 → 8192 → 1024).

    Attention is computed over the FIXED 33-length static KV with an additive
    causal/length mask — shape-static (CG-ready); at 33 positions the
    masked-full compute is cheaper than any varlen cleverness.
    """

    def __init__(self, depth_cfg: Any, layer_idx: int) -> None:
        """Dense q/k/v/o + gate/up/down projections + 2 RMSNorms; no stacking
        (own modules so the manual weight copies stay 1:1)."""
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = _CsmDepthAttention(depth_cfg)
        self.mlp = _CsmDepthMLP(depth_cfg)
        self.input_layernorm = CsmRMSNorm(
            depth_cfg.hidden_size, eps=depth_cfg.rms_norm_eps
        )
        self.post_attention_layernorm = CsmRMSNorm(
            depth_cfg.hidden_size, eps=depth_cfg.rms_norm_eps
        )

    def forward(
        self,
        hidden_BTD: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        write_pos: int,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """One layer forward over the static 33-position KV.

        Args:
            hidden_BTD: ``[B, T, 1024]`` bf16; T == 2 on the len-2 "depth
                prefill" (step 0+1), T == 1 thereafter.
            rope_cos / rope_sin: precomputed ``[33, 128]`` fp32 cos/sin slices
                for the positions being written.
            k_cache / v_cache: this layer's static-KV slot views
                ``[B, 2, 33, 128]`` bf16 (kv-heads-major), written in place at
                ``write_pos``.
            write_pos: first absolute depth position (0..32) being written.
            attn_mask: additive ``[T, 33]`` (or broadcastable) fp mask encoding
                causal + valid-length.

        Returns:
            ``[B, T, 1024]`` bf16.
        """
        residual = hidden_BTD
        hidden = self.input_layernorm(hidden_BTD)
        hidden = self.self_attn(
            hidden, rope_cos, rope_sin, k_cache, v_cache, write_pos, attn_mask
        )
        hidden = residual + hidden

        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        return residual + hidden


class CsmDepthDecoder(nn.Module):
    """Dense depth decoder: 31-step inner AR per 80 ms frame.

    Owns ``inputs_embeds_projector [1024, 2048]`` (Linear 2048→1024, no bias),
    4 :class:`CsmDepthDecoderLayer`, final RMSNorm, a
    :class:`CsmCodebooksHead`, precomputed 33-position rope cos/sin, and the
    SLOT-INDEXED static KV ``k, v: [4, max_slots, 2, 33, 128]`` bf16.

    Slot- (not pool-) indexed by design: the depth cache is
    frame-local — reset at the start of every frame, dead by the end — so it
    needs no per-request persistence and no rid→row gather/scatter; it lives
    in CG-batch-slot order ``[:bs]``. "Reset per frame" = nothing at all: the
    causal rows of the precomputed mask never look past ``write_pos + T - 1``,
    so stale KV from the previous frame is unreachable (no memset, no length
    scalar).
    """

    def __init__(
        self,
        depth_cfg: Any,
        frame_embedding: CsmFrameEmbedding,
        max_slots: int,
    ) -> None:
        """``frame_embedding`` is SHARED with the outer model (tied table);
        ``max_slots`` sizes the static KV (= CG max batch size)."""
        super().__init__()
        self.config = depth_cfg
        self.max_slots = max_slots
        self.num_codebooks = depth_cfg.num_codebooks
        self.frame_embedding = frame_embedding
        self.inputs_embeds_projector = nn.Linear(
            depth_cfg.backbone_hidden_size, depth_cfg.hidden_size, bias=False
        )
        self.layers = nn.ModuleList(
            CsmDepthDecoderLayer(depth_cfg, i)
            for i in range(depth_cfg.num_hidden_layers)
        )
        self.norm = CsmRMSNorm(depth_cfg.hidden_size, eps=depth_cfg.rms_norm_eps)
        self.codebooks_head = CsmCodebooksHead(
            depth_cfg.num_codebooks - 1, depth_cfg.hidden_size, depth_cfg.vocab_size
        )

        ctx = depth_cfg.max_position_embeddings  # 33
        inv_freq = _llama3_scaled_inv_freq(
            float(depth_cfg.rope_theta), depth_cfg.head_dim, depth_cfg.rope_scaling
        )
        freqs = torch.outer(torch.arange(ctx, dtype=torch.float32), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("rope_cos", emb.cos(), persistent=False)
        self.register_buffer("rope_sin", emb.sin(), persistent=False)

        # Additive causal mask over the full static window; row p hides every
        # key slot > p, which is also what makes per-frame KV reset a no-op.
        causal = torch.zeros(ctx, ctx, dtype=torch.float32)
        causal.masked_fill_(
            torch.ones(ctx, ctx, dtype=torch.bool).triu(1),
            torch.finfo(torch.float32).min,
        )
        self.register_buffer("attn_mask_full", causal, persistent=False)

        kv_shape = (
            depth_cfg.num_hidden_layers,
            max_slots,
            depth_cfg.num_key_value_heads,
            ctx,
            depth_cfg.head_dim,
        )
        self.register_buffer("k_cache", torch.zeros(kv_shape), persistent=False)
        self.register_buffer("v_cache", torch.zeros(kv_shape), persistent=False)

    def _forward_step(
        self,
        embeds_BT2048: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        write_pos: int,
    ) -> torch.Tensor:
        """Project + run all layers for the T positions starting at
        ``write_pos``; returns the post-final-norm hidden of the LAST position
        ``[B, 1024]``."""
        hidden = self.inputs_embeds_projector(embeds_BT2048)
        T = hidden.shape[1]
        mask = self.attn_mask_full[write_pos : write_pos + T].to(hidden.dtype)
        for i, layer in enumerate(self.layers):
            hidden = layer(
                hidden,
                self.rope_cos,
                self.rope_sin,
                k_cache[i],
                v_cache[i],
                write_pos,
                mask,
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
        """Run the 31-step depth AR for one frame, batched across ``[:bs]``.

        Step 0+1: forward ``[proj(h), proj(embed_codebook(cb0, 0))]`` (len-2
        "depth prefill") → head ``W[0]`` → sample cb1. Then 30 more len-1
        steps: position ``p`` embeds the previous code with table offset
        ``(p - 1) * 2051``, head ``W[p - 1]`` samples ``cb_p`` — total 31
        sampled codes; cb31 is never forwarded (HF parity). Fixed
        31-iteration host loop, no data-dependent branches (CG-capturable as
        one graph). Logits cast fp32 before sampling; per-row branchless
        greedy/top-k via :func:`sampler.sample_codes_batched`.

        Args:
            h_B2048: ``[B, 2048]`` bf16 post-norm backbone hidden.
            cb0_B: int64 ``[B]`` sampled codebook-0 codes.
            temps_B: fp32 ``[B]`` depth temperatures (CSM-private param set).
            top_ks_B: int64 ``[B]`` depth top-k buffer (K_MAX for neutral rows).
            bs: live CG batch size (buffers are sliced ``[:bs]``).

        Returns:
            int64 ``[B, 32]`` — full frame ``[cb0 | cb1..cb31]``.
        """
        assert bs <= self.max_slots, f"bs={bs} exceeds depth KV slots {self.max_slots}"
        assert h_B2048.shape[0] == bs and cb0_B.shape[0] == bs
        temps = temps_B[:bs]
        top_ks = top_ks_B[:bs]
        k_cache = self.k_cache[:, :bs]
        v_cache = self.v_cache[:, :bs]

        # Depth prefill, positions 0+1: position 0 is the RAW 2048-dim backbone
        # hidden (replaces the placeholder embedding, modeling_csm.py:480-481);
        # position 1 is cb0 embedded with codebook table 0. Both then pass the
        # 2048→1024 projector (modeling_csm.py:488).
        embeds = torch.stack(
            [h_B2048, self.frame_embedding.embed_codebook(cb0_B, 0)], dim=1
        )
        h_last = self._forward_step(embeds, k_cache, v_cache, write_pos=0)
        prev = sampler.sample_codes_batched(
            self.codebooks_head(h_last, 0).float(), temps, top_ks
        )
        codes = [cb0_B, prev]

        # Positions 2..31: position p holds cb_{p-1} (table offset (p-1)*2051),
        # head W[p-1] samples cb_p. cb31 (sampled at p=31) is never forwarded.
        for p in range(2, self.num_codebooks):
            embeds = self.frame_embedding.embed_codebook(prev, p - 1).unsqueeze(1)
            h_last = self._forward_step(embeds, k_cache, v_cache, write_pos=p)
            prev = sampler.sample_codes_batched(
                self.codebooks_head(h_last, p - 1).float(), temps, top_ks
            )
            codes.append(prev)

        return torch.stack(codes, dim=1)


__all__ = [
    "CsmCodebooksHead",
    "CsmDepthDecoder",
    "CsmDepthDecoderLayer",
    "CsmFrameEmbedding",
    "CsmRMSNorm",
]
