# SPDX-License-Identifier: Apache-2.0
"""Base modules for Qwen3-Omni model components."""

from __future__ import annotations

from typing import Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Position Embeddings ----


class SinusoidsPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for audio encoder."""

    def __init__(self, length: int, channels: int, max_timescale: int = 10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding requires even channels")
        log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(channels // 2, dtype=torch.float32)
        )
        scaled_time = (
            torch.arange(length, dtype=inv_timescales.dtype)[:, None]
            * inv_timescales[None, :]
        )
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int) -> torch.Tensor:
        return self.positional_embedding[:seqlen, :]


class MRoPE(nn.Module):
    """Multi-dimensional Rotary Position Embedding (M-RoPE)."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
        mrope_section: list[int] | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.mrope_section = mrope_section
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        """Apply interleaved MRoPE to 3D rotary embeddings."""
        if not self.mrope_section:
            return freqs[0]
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq = self.inv_freq.to(position_ids.device)
        inv_freq_expanded = inv_freq[None, None, :, None].float().expand(
            3, position_ids.shape[1], -1, 1
        )
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---- Attention & MLP ----


class Attention(nn.Module):
    """Multi-head attention with optional QK normalization and GQA support."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        bias: bool = False,
        rms_norm_eps: float = 1e-6,
        use_qk_norm: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)

        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_value: Any = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Any]:
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        if use_cache:
            past_key_value = (k, v)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            enable_gqa=self.num_kv_groups > 1,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


class MLP(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU() if hidden_act == "silu" else nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---- MoE Components ----


class MoeExperts(nn.ModuleList):
    """Collection of MoE experts with checkpoint-compatible naming."""

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        moe_intermediate_size: int,
        hidden_act: str = "silu",
    ):
        super().__init__([
            MLP(hidden_size, moe_intermediate_size, hidden_act)
            for _ in range(num_experts)
        ])
        self.num_experts = num_experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            current_hidden = self[expert_idx](current_state)
            current_hidden = current_hidden * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden)

        return final_hidden_states


class TopKRouter(nn.Module):
    """Router for selecting top-k experts."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = F.linear(hidden_states, self.weight)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        top_k_weights, top_k_index = torch.topk(
            routing_weights, self.top_k, dim=-1, sorted=False
        )
        if self.norm_topk_prob:
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(hidden_states.dtype)
        return router_logits, top_k_weights, top_k_index


class SparseMoeBlock(nn.Module):
    """Sparse MoE block combining router and experts."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        shared_expert_intermediate_size: int = 0,
        hidden_act: str = "silu",
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        self.experts = MoeExperts(
            num_experts=num_experts,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            hidden_act=hidden_act,
        )
        self.gate = TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            norm_topk_prob=norm_topk_prob,
        )
        self.shared_expert: MLP | None = None
        self.shared_expert_gate: nn.Linear | None = None
        if shared_expert_intermediate_size > 0:
            self.shared_expert = MLP(
                hidden_size=hidden_size,
                intermediate_size=shared_expert_intermediate_size,
                hidden_act=hidden_act,
            )
            self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = (
            self.shared_expert(hidden_states_reshaped)
            if self.shared_expert is not None
            else None
        )
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        final_hidden_states = self.experts(
            hidden_states_reshaped, selected_experts, routing_weights
        )
        if shared_expert_output is not None:
            shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states_reshaped))
            final_hidden_states = final_hidden_states + shared_gate * shared_expert_output
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


# ---- Decoder Layers ----


class DecoderLayer(nn.Module):
    """Standard transformer decoder layer with attention and MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        use_qk_norm: bool = False,
    ):
        super().__init__()
        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            use_qk_norm=use_qk_norm,
        )
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_value: Any = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Any]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, past_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, past_key_value


class MoeDecoderLayer(nn.Module):
    """Decoder layer with MoE (Mixture of Experts) MLP."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        shared_expert_intermediate_size: int = 0,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        hidden_act: str = "silu",
        norm_topk_prob: bool = True,
        use_qk_norm: bool = False,
    ):
        super().__init__()
        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            use_qk_norm=use_qk_norm,
        )
        self.mlp = SparseMoeBlock(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            hidden_act=hidden_act,
            norm_topk_prob=norm_topk_prob,
        )
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_value: Any = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Any]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, past_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, past_key_value


# ---- Audio Encoder Components ----


class AudioEncoderAttention(nn.Module):
    """Self-attention for audio encoder layers."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.embed_dim = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_proj(attn_output)


class AudioEncoderLayer(nn.Module):
    """Single layer of audio encoder with attention and FFN."""

    def __init__(
        self,
        d_model: int,
        encoder_attention_heads: int,
        encoder_ffn_dim: int,
        activation_function: str = "gelu",
    ):
        super().__init__()
        self.embed_dim = d_model
        self.self_attn = AudioEncoderAttention(d_model, encoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, encoder_ffn_dim)
        self.fc2 = nn.Linear(encoder_ffn_dim, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.activation_fn = nn.GELU() if activation_function == "gelu" else nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---- Code2Wav Components ----


class SnakeBeta(nn.Module):
    """Snake activation with learnable alpha and beta parameters.

    SnakeBeta := x + 1/exp(beta) * sin^2(x * exp(alpha))
    """

    def __init__(self, in_features: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(in_features))
        self.beta = nn.Parameter(torch.zeros(in_features))
        self.no_div_by_zero = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        return x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)


class Code2WavLayerScale(nn.Module):
    """Layer scale for pre_transformer - scales the residual connection."""

    def __init__(self, dim: int, initial_scale: float = 0.01):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim) * initial_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class Code2WavRotaryEmbedding(nn.Module):
    """Simple rotary embedding for Code2Wav (standard RoPE, no MRoPE)."""

    def __init__(self, dim: int, max_position_embeddings: int = 8000, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(x.device)
        freqs = torch.einsum("bs,d->bsd", position_ids.float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class Code2WavPreTransformerLayer(nn.Module):
    """Single transformer layer for Code2Wav pre_transformer."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rms_norm_eps: float = 1e-5,
        layer_scale_initial_scale: float = 0.01,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads

        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.self_attn.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.self_attn.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.self_attn.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)

        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.mlp.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.mlp.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.self_attn_layer_scale = Code2WavLayerScale(hidden_size, layer_scale_initial_scale)
        self.mlp_layer_scale = Code2WavLayerScale(hidden_size, layer_scale_initial_scale)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        batch_size, seq_len, _ = hidden_states.shape
        q = self.self_attn.q_proj(hidden_states)
        k = self.self_attn.k_proj(hidden_states)
        v = self.self_attn.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

        if self.num_key_value_heads < self.num_attention_heads:
            n_rep = self.num_attention_heads // self.num_key_value_heads
            k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(batch_size, self.num_attention_heads, seq_len, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(batch_size, self.num_attention_heads, seq_len, self.head_dim)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        attn_output = self.self_attn.o_proj(attn_output)

        hidden_states = residual + self.self_attn_layer_scale(attn_output)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        gate = F.silu(self.mlp.gate_proj(hidden_states))
        up = self.mlp.up_proj(hidden_states)
        mlp_output = self.mlp.down_proj(gate * up)

        hidden_states = residual + self.mlp_layer_scale(mlp_output)

        return hidden_states


class Code2WavPreTransformer(nn.Module):
    """Pre-transformer for Code2Wav - processes codec embeddings before HiFi-GAN decoder."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden_size = config.get("hidden_size", 1024)
        intermediate_size = config.get("intermediate_size", 3072)
        num_attention_heads = config.get("num_attention_heads", 16)
        num_key_value_heads = config.get("num_key_value_heads", 16)
        num_hidden_layers = config.get("num_hidden_layers", 8)
        rms_norm_eps = config.get("rms_norm_eps", 1e-5)
        layer_scale_initial_scale = config.get("layer_scale_initial_scale", 0.01)
        max_position_embeddings = config.get("max_position_embeddings", 8000)
        rope_theta = config.get("rope_theta", 10000.0)
        head_dim = hidden_size // num_attention_heads

        self.layers = nn.ModuleList([
            Code2WavPreTransformerLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                rms_norm_eps=rms_norm_eps,
                layer_scale_initial_scale=layer_scale_initial_scale,
            )
            for _ in range(num_hidden_layers)
        ])
        self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = Code2WavRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings)
        return self.norm(hidden_states)


class CausalConv1d(nn.Module):
    """Causal 1D convolution with proper padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation
        self.padding = self.kernel_size - self.stride

    def _get_extra_padding_for_conv1d(self, x: torch.Tensor) -> int:
        length = x.shape[-1]
        n_frames = (length - self.kernel_size + self.padding) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (self.kernel_size - self.padding)
        return ideal_length - length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extra_padding = self._get_extra_padding_for_conv1d(x)
        x = F.pad(x, (self.padding, extra_padding), mode="constant", value=0)
        return self.conv(x)


class CausalTransConv1d(nn.Module):
    """Causal transposed 1D convolution for upsampling."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)
        pad = kernel_size - stride
        self.left_pad = math.ceil(pad)
        self.right_pad = self.left_pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x[..., self.left_pad : x.shape[-1] - self.right_pad]


class Code2WavConvNeXtBlock(nn.Module):
    """ConvNeXt-style block used in Code2Wav upsampling."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = CausalConv1d(dim, dim, kernel_size=7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 2, 1)
        return residual + x


class Code2WavResidualUnit(nn.Module):
    """Residual unit with Snake activations and dilated convolutions."""

    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        self.act1 = SnakeBeta(dim)
        self.conv1 = CausalConv1d(dim, dim, kernel_size=7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = CausalConv1d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act1(x)
        x = self.conv1(x)
        x = self.act2(x)
        x = self.conv2(x)
        return x + residual


class Code2WavDecoderBlock(nn.Module):
    """Decoder block with upsampling and residual units."""

    def __init__(self, in_dim: int, out_dim: int, upsample_rate: int):
        super().__init__()
        self.block = nn.ModuleList([
            SnakeBeta(in_dim),
            CausalTransConv1d(in_dim, out_dim, 2 * upsample_rate, upsample_rate),
            Code2WavResidualUnit(out_dim, dilation=1),
            Code2WavResidualUnit(out_dim, dilation=3),
            Code2WavResidualUnit(out_dim, dilation=9),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.block:
            x = block(x)
        return x
