"""
FishQwen3 models following HuggingFace conventions.

This module contains the model implementations:
- FishQwen3Model: Base transformer without any head
- FishQwen3ForCausalLM: For language modeling
- FishQwen3OmniForCausalLM: Omni model for causal language modeling with audio
"""

import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from sgl_kernel.flash_attn import flash_attn_with_kvcache

# liger_kernel removed for inference
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel

from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.configuration import (
    FishQwen3AudioDecoderConfig,
    FishQwen3Config,
    FishQwen3OmniConfig,
)
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.utils import (
    apply_rotary_emb,
    find_multiple,
    precompute_freqs_cis,
)
from sglang_omni.models.fishaudio_s2_pro.fish_speech.utils import RankedLogger

flash_attn_varlen_func = None
FLASH_ATTN_VERSION = 2

log = RankedLogger(__name__, rank_zero_only=True)

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
    """RMSNorm layer."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms = norm_x * (self.dim**-0.5)
        x_normed = x / (rms + self.eps)
        return x_normed * self.weight


RMSNorm = MyRMSNorm if FISH_BATCH_INVARIANT else nn.RMSNorm


@dataclass
class FishQwen3OmniOutput:
    """Output for Dual-AR models."""

    token_logits: Optional[Tensor] = None
    token_weights: Optional[Tensor] = None
    codebook_logits: Optional[Tensor] = None
    codebook_weights: Optional[Tensor] = None
    token_hidden_states: Optional[Tensor] = None
    codebook_hidden_states: Optional[Tensor] = None
    router_logits: Optional[Tensor] = None
    # expert_indices: Tuple of (seq_len, top_k) tensors, one per MoE layer
    # Contains the indices of the top-k experts selected for each token in each layer
    # Used for MoE routing replay
    expert_indices: Optional[tuple] = None


# ============================================================================
# Building blocks (shared across all models)
# ============================================================================


class Attention(nn.Module):
    """Multi-head attention with optional GQA and RoPE."""

    def __init__(self, config: FishQwen3Config):
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

        self.dropout = config.dropout
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.attention_qk_norm = config.attention_qk_norm

        # KV cache for generation (initialized by setup_caches)
        self.kv_cache: Optional["KVCache"] = None

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        """Handle legacy checkpoint format."""
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        cumsum_lengths: Optional[Tensor] = None,
        max_length: Optional[int] = None,
    ) -> Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor (batch_size, seq_len, dim) or (seq_len, dim)
            freqs_cis: RoPE frequencies (seq_len, head_dim)
            cumsum_lengths: Cumulative sequence lengths (optional)
            max_length: Maximum sequence length (optional)
        """

        if x.ndim == 2:
            seqlen, _ = x.shape
            bsz = 1
        else:
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

        if cumsum_lengths is not None:
            assert bsz == 1, "Cumsum lengths only supported in single sample mode"
            q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)

            kwargs = {}
            if FLASH_ATTN_VERSION == 3 and FISH_BATCH_INVARIANT:
                kwargs["num_splits"] = True

            # Force cast to bfloat16 for flash attention
            y = flash_attn_varlen_func(
                q=q.to(torch.bfloat16),
                k=k.to(torch.bfloat16),
                v=v.to(torch.bfloat16),
                cu_seqlens_q=cumsum_lengths,
                cu_seqlens_k=cumsum_lengths,
                max_seqlen_q=max_length,
                max_seqlen_k=max_length,
                causal=True,
                deterministic=FISH_BATCH_INVARIANT,
                **kwargs,
            )
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

            y = self._scaled_dot_product_attention(q, k, v)
            y = y.transpose(1, 2)

        if isinstance(y, tuple):
            y = y[0]

        y = y.contiguous()

        if x.ndim == 2:
            y = y.view(seqlen, q_size)
        else:
            y = y.view(bsz, seqlen, q_size)

        return self.wo(y)

    def _scaled_dot_product_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = True,
    ) -> Tensor:
        """Fallback attention implementation."""
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(1, 1, L, S, dtype=query.dtype, device=query.device)

        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(
                diagonal=0
            )
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

        return attn_weight @ value

    def forward_kvcached(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        cache_seqlens: Tensor,
    ) -> Tensor:
        """
        Forward pass with KV cache using flash_attn_with_kvcache.

        Args:
            x: Input tensor (batch_size, seq_len, dim)
            freqs_cis: RoPE frequencies (seq_len, head_dim)
            cache_seqlens: Current sequence lengths in the KV cache (batch_size,)

        Returns:
            Output tensor (batch_size, seq_len, dim)
        """

        bsz, seqlen, _ = x.shape

        q_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([q_size, kv_size, kv_size], dim=-1)

        # Shape: (batch_size, seqlen, n_heads, head_dim)
        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if self.attention_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Use flash_attn_with_kvcache - it handles KV cache update internally
        # q: (batch_size, seqlen, nheads, headdim)
        # k_cache/v_cache: (batch_size, seqlen_cache, nheads_k, headdim)
        # k/v: (batch_size, seqlen_new, nheads_k, headdim)
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

        y = y.contiguous().view(bsz, seqlen, q_size)

        return self.wo(y)


class KVCache(nn.Module):
    """
    KV cache for flash_attn_with_kvcache.

    Cache shape: (batch_size, max_seq_len, n_heads, head_dim)
    This matches the expected format for flash_attn_with_kvcache.
    """

    def __init__(
        self, max_batch_size, max_seq_len, n_heads, head_dim, dtype=torch.bfloat16
    ):
        super().__init__()
        # Shape: (batch_size, seqlen_cache, nheads_k, headdim)
        cache_shape = (max_batch_size, max_seq_len, n_heads, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))
        self.max_seq_len = max_seq_len

    def get(self, batch_size):
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
        if FISH_BATCH_INVARIANT is False:
            return self.w2(F.silu(x1) * x3)

        return self.w2(F.silu(x1) * x3)


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward."""

    def __init__(self, config: FishQwen3Config) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(
            dim=config.dim, intermediate_size=config.intermediate_size
        )
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        cumsum_lengths: Optional[Tensor] = None,
        max_length: Optional[int] = None,
    ) -> Tensor:
        h = x + self.attention(
            self.attention_norm(x),
            freqs_cis=freqs_cis,
            cumsum_lengths=cumsum_lengths,
            max_length=max_length,
        )
        return h + self.feed_forward(self.ffn_norm(h))

    def forward_kvcached(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        cache_seqlens: Tensor,
        **kwargs,
    ) -> Tensor:
        h = x + self.attention.forward_kvcached(
            self.attention_norm(x),
            freqs_cis=freqs_cis,
            cache_seqlens=cache_seqlens,
        )
        return h + self.feed_forward(self.ffn_norm(h))


# ============================================================================
# Pre-trained model base class
# ============================================================================


class FishQwen3PreTrainedModel(PreTrainedModel):
    """Base class for all FishQwen3 models."""

    config_class = FishQwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]

    def _init_weights(self, module):
        """Initialize weights."""
        # Handle nested configs (like FishQwen3OmniConfig)
        if hasattr(self.config, "text_config"):
            std = self.config.text_config.initializer_range
        else:
            std = self.config.initializer_range

        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(
                module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(
                module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std
            )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    @staticmethod
    def get_wrap_policy() -> set:
        """Get the default wrapping policy for FSDP."""

        return set([TransformerBlock])


# ============================================================================
# FishQwen3Model - Base model without head
# ============================================================================


class FishQwen3Model(FishQwen3PreTrainedModel):
    """
    Base FishQwen3 model without any head.

    This outputs the hidden states from the transformer.
    """

    def __init__(self, config: FishQwen3Config):
        super().__init__(config)
        self.config = config

        # Token embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.dim)

        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        # RoPE frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.max_seq_len,
                config.head_dim,
                config.rope_base,
            ),
            persistent=False,
        )

        # For KV cache tracking
        self.max_batch_size = -1
        self.max_seq_len = -1

        self.post_init()

    def setup_caches(
        self,
        max_batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Setup KV caches for efficient autoregressive generation.

        Args:
            max_batch_size: Maximum batch size
            max_seq_len: Maximum sequence length
            dtype: Data type for the cache
        """
        if self.max_seq_len >= max_seq_len and self.max_batch_size >= max_batch_size:
            return

        max_seq_len = find_multiple(max_seq_len, 8)
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

        # Get device from model parameters
        device = next(self.parameters()).device

        for layer in self.layers:
            layer.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_len,
                self.config.n_local_heads,
                self.config.head_dim,
                dtype=dtype,
            ).to(device)

    def reset_caches(self):
        """Reset all KV caches to zeros."""
        for layer in self.layers:
            if layer.attention.kv_cache is not None:
                layer.attention.kv_cache.k_cache.zero_()
                layer.attention.kv_cache.v_cache.zero_()

    def forward_kvcached(
        self,
        input_ids: Tensor,
        input_pos: Tensor,
        input_embeds: Optional[Tensor] = None,
        expert_indices: Optional[tuple] = None,
    ) -> Tensor | tuple[Tensor, tuple, tuple]:
        """
        Forward pass with KV cache for efficient autoregressive generation.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            input_pos: Position indices (seq_len,)
            input_embeds: Optional pre-computed embeddings (batch_size, seq_len, dim)
            expert_indices: Optional tuple of expert indices for MoE replay,
                            one (batch_size * seq_len, top_k) tensor per layer

        Returns:
            Hidden states (batch_size, seq_len, dim)
            If use_moe, returns tuple of (hidden_states, router_logits, expert_indices)
        """
        if input_embeds is not None:
            x = input_embeds
        else:
            x = self.embeddings(input_ids)

        bsz = x.shape[0]

        # Get RoPE frequencies for current positions
        freqs_cis = self.freqs_cis[input_pos]

        # Compute cache_seqlens from input_pos
        # input_pos contains the position indices, cache_seqlens should be the starting position
        # For a single decode step, input_pos is [pos] and cache_seqlens should be [pos]
        cache_seqlens = input_pos[0].expand(bsz).to(torch.int32)

        # Apply transformer layers
        router_logits = tuple() if self.config.use_moe else None
        expert_indices_out = tuple() if self.config.use_moe else None
        for layer_idx, layer in enumerate(self.layers):
            # Get expert indices for this layer if provided
            layer_expert_indices = None
            if expert_indices is not None and layer_idx < len(expert_indices):
                layer_expert_indices = expert_indices[layer_idx]

            result = layer.forward_kvcached(
                x, freqs_cis, cache_seqlens, expert_indices=layer_expert_indices
            )

            if self.config.use_moe:
                x, layer_router_logits, layer_expert_indices_out = result
                router_logits += (layer_router_logits,)
                expert_indices_out += (layer_expert_indices_out,)
            else:
                x = result

        x = self.norm(x)

        if self.config.use_moe:
            return x, router_logits, expert_indices_out
        return x

    def forward(
        self,
        *,
        lengths: Tensor,
        max_length: Optional[int] = None,
        cumsum_lengths: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        input_embeds: Optional[Tensor] = None,
        expert_indices: Optional[tuple] = None,
    ) -> Tensor | tuple[Tensor, tuple, tuple]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (seq_len,)
            lengths: Sequence lengths (batch_size,)
            max_length: Maximum sequence length (optional)
            cumsum_lengths: Cumulative sequence lengths (optional)
            position_ids: Position IDs (optional)
            input_embeds: Optional embeddings to replace values in x (shape matches masked positions)
            expert_indices: Optional tuple of expert indices for MoE replay,
                            one (seq_len, top_k) tensor per layer

        Returns:
            Hidden states of shape (seq_len, hidden_size)
            If use_moe, returns tuple of (hidden_states, router_logits, expert_indices)
        """

        if input_embeds is not None:
            x = input_embeds
        else:
            # Embed tokens
            x = self.embeddings(input_ids)

        x.requires_grad_(True)

        # Prepare position info
        if max_length is None:
            max_length = lengths.max().item()

        if cumsum_lengths is None:
            cumsum_lengths = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int, device=lengths.device),
                    lengths.cumsum(dim=0),
                ]
            ).int()

        if position_ids is None:
            position_ids = torch.cat(
                [torch.arange(i, dtype=torch.int) for i in lengths]
            ).to(lengths.device)

        freqs_cis = self.freqs_cis[position_ids]

        # Apply transformer layers
        router_logits = tuple() if self.config.use_moe else None
        expert_indices_out = tuple() if self.config.use_moe else None
        for layer_idx, layer in enumerate(self.layers):
            # Get expert indices for this layer if provided
            layer_expert_indices = None
            if expert_indices is not None and layer_idx < len(expert_indices):
                layer_expert_indices = expert_indices[layer_idx]

            if self.config.use_gradient_checkpointing and self.training:
                result = checkpoint(
                    layer,
                    x,
                    freqs_cis,
                    cumsum_lengths,
                    max_length,
                    layer_expert_indices,
                    use_reentrant=True,
                )
            else:
                result = layer(
                    x,
                    freqs_cis,
                    cumsum_lengths,
                    max_length,
                    expert_indices=layer_expert_indices,
                )

            # Handle MoE router logits and expert indices
            if self.config.use_moe:
                x, layer_router_logits, layer_expert_indices_out = result
                router_logits += (layer_router_logits,)
                expert_indices_out += (layer_expert_indices_out,)
            else:
                x = result

        x = self.norm(x)

        # Return router_logits and expert_indices if using MoE, otherwise just hidden states
        if self.config.use_moe:
            return x, router_logits, expert_indices_out
        return x

    def resize_token_embeddings(
        self,
        new_num_tokens: int,
        pad_to_multiple_of: Optional[int] = 8,
    ) -> nn.Embedding:
        """Resize token embeddings."""
        if pad_to_multiple_of is not None:
            new_num_tokens = find_multiple(new_num_tokens, pad_to_multiple_of)

        old_embeddings = self.embeddings
        new_embeddings = nn.Embedding(new_num_tokens, self.config.dim)
        old_num_tokens = old_embeddings.weight.shape[0]
        assert new_num_tokens > old_num_tokens

        # Initialize new embeddings with mean of old ones
        old_embeddings_weight = old_embeddings.weight.data.to(torch.float32)
        mean_embeddings = torch.mean(old_embeddings_weight, axis=0)
        old_centered_embeddings = old_embeddings_weight - mean_embeddings
        covariance = (
            old_centered_embeddings.T @ old_centered_embeddings / old_num_tokens
        )

        # Check if covariance is positive definite
        eigenvalues = torch.linalg.eigvals(covariance)
        is_covariance_psd = bool(
            (covariance == covariance.T).all()
            and not torch.is_complex(eigenvalues)
            and (eigenvalues > 0).all()
        )

        if is_covariance_psd:
            distribution = torch.distributions.multivariate_normal.MultivariateNormal(
                mean_embeddings, covariance_matrix=1e-9 * covariance
            )
            new_embeddings.weight.data[-1 * (new_num_tokens - old_num_tokens) :, :] = (
                distribution.sample(sample_shape=(new_num_tokens - old_num_tokens,)).to(
                    old_embeddings.weight.dtype
                )
            )
        else:
            new_embeddings.weight.data[-1 * (new_num_tokens - old_num_tokens) :, :] = (
                mean_embeddings[None, :]
                .repeat(new_num_tokens - old_num_tokens, 1)
                .to(old_embeddings.weight.dtype)
            )

        new_embeddings.weight.data[:old_num_tokens, :] = old_embeddings.weight.data[
            :old_num_tokens, :
        ]

        self.embeddings.weight.data = new_embeddings.weight.data
        self.embeddings.num_embeddings = new_embeddings.num_embeddings
        self.config.vocab_size = new_num_tokens

        return self.embeddings


# ============================================================================
# FishQwen3ForCausalLM - For language modeling
# ============================================================================


@dataclass
class FishQwen3CausalLMOutput:
    logits: Optional[Tensor] = None
    hidden_states: Optional[Tensor] = None
    weights: Optional[Tensor] = None
    router_logits: Optional[Tensor] = None
    # expert_indices: Tuple of (seq_len, top_k) or (batch_size * seq_len, top_k) tensors,
    # one per MoE layer, containing the indices of the top-k experts selected for each token
    expert_indices: Optional[tuple] = None


class FishQwen3ForCausalLM(FishQwen3PreTrainedModel):
    """FishQwen3 model for causal language modeling."""

    def __init__(self, config: FishQwen3Config):
        super().__init__(config)
        self.model = FishQwen3Model(config)

        # LM head
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        if self.config.tie_word_embeddings:
            return self.model.embeddings
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        if not self.config.tie_word_embeddings:
            self.lm_head = new_embeddings

    def setup_caches(
        self,
        max_batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Setup KV caches for efficient autoregressive generation."""
        self.model.setup_caches(max_batch_size, max_seq_len, dtype)

    def reset_caches(self):
        """Reset all KV caches to zeros."""
        self.model.reset_caches()

    def forward_kvcached(
        self,
        input_ids: Tensor,
        input_pos: Tensor,
        input_embeds: Optional[Tensor] = None,
        expert_indices: Optional[tuple] = None,
    ) -> FishQwen3CausalLMOutput:
        """
        Forward pass with KV cache for efficient autoregressive generation.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            input_pos: Position indices (seq_len,)
            input_embeds: Optional pre-computed embeddings (batch_size, seq_len, dim)
            expert_indices: Optional tuple of expert indices for MoE replay,
                            one (batch_size * seq_len, top_k) tensor per layer

        Returns:
            FishQwen3CausalLMOutput with logits, hidden states, and expert_indices
        """
        result = self.model.forward_kvcached(
            input_ids, input_pos, input_embeds, expert_indices=expert_indices
        )

        if self.config.use_moe:
            hidden_states, router_logits, expert_indices_out = result
        else:
            hidden_states = result
            router_logits = None
            expert_indices_out = None

        # Compute logits
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden_states, self.model.embeddings.weight)
        else:
            logits = self.lm_head(hidden_states)

        return FishQwen3CausalLMOutput(
            logits=logits,
            hidden_states=hidden_states,
            weights=None,
            router_logits=router_logits,
            expert_indices=expert_indices_out,
        )

    def forward(
        self,
        labels: Optional[Tensor] = None,
        return_weights: bool = False,
        **kwargs,
    ) -> FishQwen3CausalLMOutput:
        """Forward pass with optional loss computation."""
        # Get hidden states (and router_logits, expert_indices if MoE)
        result = self.model(**kwargs)

        if self.config.use_moe:
            hidden_states, router_logits, expert_indices_out = result
        else:
            hidden_states = result
            router_logits = None
            expert_indices_out = None

        # Get weight matrix for output projection
        if self.config.tie_word_embeddings:
            weights = self.model.embeddings.weight
        else:
            weights = self.lm_head.weight

        # Compute logits for inference/evaluation
        # Only compute if not training
        logits = None
        if return_weights is False:
            if self.config.tie_word_embeddings:
                logits = F.linear(hidden_states, self.model.embeddings.weight)
            else:
                logits = self.lm_head(hidden_states)

        return FishQwen3CausalLMOutput(
            logits=logits,
            weights=weights,
            hidden_states=hidden_states,
            router_logits=router_logits,
            expert_indices=expert_indices_out,
        )


# ============================================================================
# FishQwen3AudioDecoder - Fast decoder for codebook prediction
# ============================================================================


@dataclass
class FishQwen3AudioDecoderOutput:
    logits: Optional[Tensor] = None
    weights: Optional[Tensor] = None
    hidden_states: Optional[Tensor] = None


class FishQwen3AudioDecoder(PreTrainedModel):
    """Fast decoder for predicting audio codebooks in Dual-AR model."""

    config_class = FishQwen3AudioDecoderConfig
    base_model_prefix = "audio_decoder"

    def __init__(self, config: FishQwen3AudioDecoderConfig):
        super().__init__(config)
        self.config = config

        # Project from model dim to fast dim if needed
        if config.text_dim != config.dim:
            self.project_in = nn.Linear(config.text_dim, config.dim)
        else:
            self.project_in = nn.Identity()

        # Codebook embeddings for VQ tokens in text model
        self.codebook_embeddings = nn.Embedding(
            config.vocab_size * config.num_codebooks,
            config.text_dim,
        )

        # Codebook embeddings for fast decoder
        self.embeddings = nn.Embedding(config.vocab_size, config.dim)

        # Fast decoder transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # RoPE frequencies for fast decoder
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.num_codebooks,
                config.head_dim,
                config.rope_base,
            ),
            persistent=False,
        )

        # Codebook offsets for vectorized embedding lookup
        self.register_buffer(
            "codebook_offsets",
            torch.arange(config.num_codebooks) * config.vocab_size,
            persistent=False,
        )

        # For KV cache tracking
        self.max_batch_size = -1

    def setup_caches(
        self,
        max_batch_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Setup KV caches for efficient autoregressive codebook generation."""
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

        # Pre-allocate input_pos buffer for CUDA graph compatibility
        # This avoids creating tensors during generation
        self.register_buffer(
            "input_pos",
            torch.zeros(1, device=device, dtype=torch.long),
            persistent=False,
        )

    def reset_caches(self):
        """Reset all KV caches to zeros."""
        for layer in self.layers:
            if layer.attention.kv_cache is not None:
                layer.attention.kv_cache.k_cache.zero_()
                layer.attention.kv_cache.v_cache.zero_()

    def forward_kvcached(
        self,
        x: Tensor,
        codebook_idx: int,
    ) -> Tensor:
        """
        Forward pass with KV cache for efficient autoregressive codebook generation.

        Args:
            x: Input embeddings (batch_size, 1, dim)
            codebook_idx: Position index as integer (0 to num_codebooks)

        Returns:
            Logits tensor (batch_size, 1, vocab_size)
        """
        bsz = x.shape[0]

        # Update the pre-allocated input_pos buffer (for CUDA graph compatibility)
        self.input_pos.fill_(codebook_idx)

        freqs_cis = self.freqs_cis[self.input_pos]

        # cache_seqlens: current position in cache for each batch item
        cache_seqlens = self.input_pos.expand(bsz).to(torch.int32)

        for layer in self.layers:
            x = layer.forward_kvcached(x, freqs_cis, cache_seqlens)

        x = self.norm(x)
        return self.output(x)

    def embed_text_dim(
        self,
        x: Tensor,
        vq_parts: Optional[Tensor] = None,
        vq_mask_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Combine text embeddings with VQ codebook embeddings.

        Args:
            x: Text embeddings (batch_size, seq_len, text_dim)
            vq_parts: VQ codebook IDs (num_masked, num_codebooks)
            vq_mask_tokens: Boolean mask for VQ positions (batch_size, seq_len)

        Returns:
            Combined embeddings with VQ tokens added at masked positions
        """
        if vq_parts is None or vq_mask_tokens is None:
            return x

        # Vectorized: add offsets to all codebooks at once
        offset_parts = (
            vq_parts + self.codebook_offsets[None, :]
        )  # (num_masked, num_codebooks)
        # Embed all codebooks at once
        all_embeds = self.codebook_embeddings(
            offset_parts
        )  # (num_masked, num_codebooks, text_dim)
        # Sum over codebooks
        vq_embeds_sum = all_embeds.sum(dim=1)  # (num_masked, text_dim)

        vq_summed_embeds = x[vq_mask_tokens] + vq_embeds_sum.to(x.dtype)

        return vq_summed_embeds / math.sqrt(self.config.num_codebooks + 1)

    def embed_one_token(
        self,
        text_embeds: Tensor,
        vq_parts: Tensor,
        is_semantic: Tensor,
    ) -> Tensor:
        """
        Combine text embeddings with VQ codebook embeddings for a single decode step.

        This is used during autoregressive generation to embed the previous token
        before feeding it to the model.

        Args:
            text_embeds: Text embeddings (batch_size, dim) - squeezed, no seq dim
            vq_parts: VQ codebook values (batch_size, num_codebooks)
            is_semantic: Boolean mask indicating semantic tokens (batch_size,)

        Returns:
            Combined embeddings (batch_size, dim) with VQ added for semantic tokens
        """
        # Vectorized: add offsets to all codebooks at once
        offset_parts = (
            vq_parts + self.codebook_offsets[None, :]
        )  # (batch_size, num_codebooks)
        # Embed all codebooks at once
        all_embeds = self.codebook_embeddings(
            offset_parts
        )  # (batch_size, num_codebooks, dim)
        # Sum over codebooks
        vq_embeds_sum = all_embeds.sum(dim=1)  # (batch_size, dim)

        # Combine text and VQ embeddings for semantic tokens
        combined = (text_embeds + vq_embeds_sum.to(text_embeds.dtype)) / math.sqrt(
            self.config.num_codebooks + 1
        )

        # Use where to select: combined for semantic, original for non-semantic
        # is_semantic should be (batch_size,), unsqueeze to (batch_size, 1) for proper broadcasting
        return torch.where(is_semantic.unsqueeze(-1), combined, text_embeds)

    def forward(
        self,
        hidden_states: Tensor,
        vq_mask_labels: Tensor,
        vq_require_losses: Tensor,
        vq_parts: Tensor,
        return_weights: bool = False,
    ) -> Optional[FishQwen3AudioDecoderOutput]:
        """
        Forward pass for fast audio decoder.

        Args:
            hidden_states: Hidden states from base model (batch_size, seq_len, hidden_dim)
            vq_mask_labels: Boolean mask for positions requiring codebook prediction
            vq_require_losses: Boolean mask for which positions need loss computation
            vq_parts: Ground truth codebooks (num_masked, num_codebooks)

        Returns:
            Codebook logits of shape (num_required, num_codebooks+1, codebook_size) or None
        """
        if vq_mask_labels is None or vq_require_losses is None:
            return None

        # Extract relevant hidden states
        x = hidden_states[vq_mask_labels][vq_require_losses]
        codebooks = vq_parts[..., :-1][vq_require_losses]

        # Project to fast dimension
        x = self.project_in(x)

        # Get codebook embeddings and prepend hidden state
        codebook_embeddings = self.embeddings(codebooks)
        x = torch.cat([x[:, None], codebook_embeddings], dim=1)
        x.requires_grad_(True)

        # Apply transformer layers
        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, self.freqs_cis, None, None, use_reentrant=True)
            else:
                x = layer(x, self.freqs_cis, None, None)

        # Output projection
        x = self.norm(x)

        if return_weights:
            return FishQwen3AudioDecoderOutput(
                logits=None, weights=self.output.weight, hidden_states=x
            )

        codebook_logits = self.output(x)
        return FishQwen3AudioDecoderOutput(
            logits=codebook_logits,
            weights=None,
            hidden_states=x,
        )


def _get_feat_extract_output_lengths(input_lengths: Tensor | int) -> Tensor | int:
    """
    Computes the output length of the convolutional layers.

    The audio encoder uses 3 Conv2d layers with stride 2, resulting in 8x downsampling.
    """
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = (
        ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    )
    return output_lengths


# ============================================================================
# FishQwen3OmniForCausalLM - Omni model with text model + audio decoder
# ============================================================================


class FishQwen3OmniForCausalLM(FishQwen3PreTrainedModel):
    """
    FishQwen3Omni model combining text model and audio decoder for multimodal tasks.

    This model is composed of separate components:
    - Text Model (FishQwen3ForCausalLM) handles language modeling
    - Optional Audio Decoder (FishQwen3AudioDecoder) for dual-AR codebook prediction
    """

    config_class = FishQwen3OmniConfig

    def __init__(self, config: FishQwen3OmniConfig):
        # Initialize with text config as base
        super().__init__(config)
        self.config = config

        # Text model component (full language model)
        self.text_model = FishQwen3ForCausalLM(config.text_config)

        # Optional audio decoder for dual-AR models
        self.audio_decoder = None
        if config.audio_decoder_config is not None:
            self.audio_decoder = FishQwen3AudioDecoder(config.audio_decoder_config)

        self.post_init()

    def setup_caches(
        self,
        max_batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Setup KV caches for efficient autoregressive generation."""
        self.text_model.setup_caches(max_batch_size, max_seq_len, dtype)
        if self.audio_decoder is not None:
            self.audio_decoder.setup_caches(max_batch_size, dtype)

    def reset_caches(self):
        """Reset all KV caches to zeros."""
        self.text_model.reset_caches()
        if self.audio_decoder is not None:
            self.audio_decoder.reset_caches()

    def expand_kv_cache(self, num_samples: int, seq_len: int):
        """
        Expand KV cache from batch slot 0 to slots 1..num_samples-1.

        This is used for efficient multi-sample generation where we do prefill
        with batch_size=1 and then expand the KV cache for parallel decoding.

        Args:
            num_samples: Number of samples to expand to
            seq_len: Sequence length of the prefill (positions 0..seq_len-1)
        """
        for layer in self.text_model.model.layers:
            if layer.attention.kv_cache is not None:
                # Copy from slot 0 to slots 1..num_samples-1
                # KV cache shape: (max_batch_size, max_seq_len, n_heads, head_dim)
                layer.attention.kv_cache.k_cache[1:num_samples, :seq_len].copy_(
                    layer.attention.kv_cache.k_cache[0:1, :seq_len].expand(
                        num_samples - 1, -1, -1, -1
                    )
                )
                layer.attention.kv_cache.v_cache[1:num_samples, :seq_len].copy_(
                    layer.attention.kv_cache.v_cache[0:1, :seq_len].expand(
                        num_samples - 1, -1, -1, -1
                    )
                )

    def forward_kvcached(
        self,
        input_ids: Tensor,
        input_pos: Tensor,
        input_embeds: Optional[Tensor] = None,
        expert_indices: Optional[tuple] = None,
    ) -> FishQwen3OmniOutput:
        """
        Forward pass with KV cache for efficient autoregressive generation.

        This simplified forward doesn't handle audio encoder or VQ embeddings,
        just does straight text generation with KV cache.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            input_pos: Position indices (seq_len,)
            input_embeds: Optional pre-computed embeddings (batch_size, seq_len, dim)
            expert_indices: Optional tuple of expert indices for MoE replay,
                            one (batch_size * seq_len, top_k) tensor per layer

        Returns:
            FishQwen3OmniOutput with token logits, hidden states, and expert_indices
        """
        # Get embeddings if not provided
        if input_embeds is None:
            input_embeds = self.text_model.model.embeddings(input_ids)

        # Forward through text model with KV cache
        text_result = self.text_model.forward_kvcached(
            input_ids=input_ids,
            input_pos=input_pos,
            input_embeds=input_embeds,
            expert_indices=expert_indices,
        )

        return FishQwen3OmniOutput(
            token_logits=text_result.logits,
            token_hidden_states=text_result.hidden_states,
            token_weights=None,
            codebook_logits=None,
            codebook_hidden_states=None,
            codebook_weights=None,
            router_logits=text_result.router_logits,
            expert_indices=text_result.expert_indices,
        )

    def embed(
        self,
        input_ids: Tensor,
        vq_parts: Optional[Tensor] = None,
        vq_mask_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        # Prepare embeddings for text model
        x = self.text_model.model.embeddings(input_ids)

        # Add VQ embeddings if provided
        if self.audio_decoder and vq_parts is not None and vq_mask_tokens is not None:
            vq_embeds = self.audio_decoder.embed_text_dim(x, vq_parts, vq_mask_tokens)
            x[vq_mask_tokens] = vq_embeds.to(x.dtype)

        return x

    def embed_one_token(
        self,
        token_ids: Tensor,
        vq_parts: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute embeddings for a single token during autoregressive decode step.

        This handles combining text embeddings with VQ codebook embeddings
        when the previous token was semantic.

        Args:
            token_ids: Token IDs (batch_size, 1)
            vq_parts: VQ codebook values (batch_size, num_codebooks) or None

        Returns:
            Combined embeddings (batch_size, 1, dim)
        """
        # Get text embeddings
        text_embeds = self.text_model.model.embeddings(
            token_ids
        )  # (batch_size, 1, dim)

        # If no audio decoder, no VQ parts, or no semantic token IDs configured, just return text embeddings
        if (
            self.audio_decoder is None
            or vq_parts is None
            or self.config.semantic_start_token_id is None
            or self.config.semantic_end_token_id is None
        ):
            return text_embeds

        # Check if the token is semantic
        is_semantic = (token_ids >= self.config.semantic_start_token_id) & (
            token_ids <= self.config.semantic_end_token_id
        )
        is_semantic = is_semantic.squeeze(-1)  # (batch_size,)

        # Combine text and VQ embeddings using audio_decoder's method
        text_embeds_squeezed = text_embeds.squeeze(1)  # (batch_size, dim)
        combined = self.audio_decoder.embed_one_token(
            text_embeds_squeezed, vq_parts, is_semantic
        )

        return combined.unsqueeze(1)  # (batch_size, 1, dim)

    def forward(
        self,
        input_ids: Tensor,
        lengths: Tensor,
        vq_parts: Optional[Tensor] = None,
        vq_mask_tokens: Optional[Tensor] = None,
        vq_mask_labels: Optional[Tensor] = None,
        vq_require_losses: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        return_weights: bool = False,
        expert_indices: Optional[tuple] = None,
        **kwargs,
    ) -> FishQwen3OmniOutput:
        """
        Forward pass for Omni model.

        Args:
            input_ids: Token IDs (seq_len,)
            lengths: Sequence lengths (batch_size,)
            vq_parts: VQ codebook IDs for audio (if using dual-AR)
            vq_mask_tokens: Mask for VQ token positions
            vq_mask_labels: Mask for VQ label positions
            vq_require_losses: Mask for positions requiring VQ loss
            labels: Target labels for language modeling
            return_weights: Whether to return output weights
            expert_indices: Optional tuple of expert indices for MoE replay,
                            one (seq_len, top_k) tensor per layer
            **kwargs: Additional arguments

        Returns:
            FishQwen3OmniOutput with token and codebook logits, and expert_indices
        """

        # Prepare embeddings for text model
        x = self.embed(
            input_ids,
            vq_parts,
            vq_mask_tokens,
        )

        # Get hidden states from text model
        text_result = self.text_model(
            lengths=lengths,
            input_embeds=x,
            return_weights=return_weights,
            labels=labels,
            expert_indices=expert_indices,
            **kwargs,
        )

        # Codebook logits from audio decoder
        if self.audio_decoder is None:
            return FishQwen3OmniOutput(
                token_logits=text_result.logits,
                token_hidden_states=text_result.hidden_states,
                token_weights=text_result.weights,
                codebook_logits=None,
                codebook_hidden_states=None,
                codebook_weights=None,
                router_logits=text_result.router_logits,
                expert_indices=text_result.expert_indices,
            )

        codebook_output = self.audio_decoder(
            hidden_states=text_result.hidden_states,
            vq_mask_labels=vq_mask_labels,
            vq_require_losses=vq_require_losses,
            vq_parts=vq_parts,
            return_weights=return_weights,
        )

        return FishQwen3OmniOutput(
            token_logits=text_result.logits,
            token_hidden_states=text_result.hidden_states,
            token_weights=text_result.weights,
            codebook_logits=codebook_output.logits if codebook_output else None,
            codebook_hidden_states=(
                codebook_output.hidden_states if codebook_output else None
            ),
            codebook_weights=codebook_output.weights if codebook_output else None,
            router_logits=text_result.router_logits,
            expert_indices=text_result.expert_indices,
        )

    def set_trainable_modules(
        self,
        *,
        text_model: bool = True,
        audio_decoder: bool = True,
    ):
        for param in self.text_model.parameters():
            param.requires_grad = text_model

        if self.audio_decoder is not None:
            for param in self.audio_decoder.parameters():
                param.requires_grad = audio_decoder

    def set_use_gradient_checkpointing(self, use_gradient_checkpointing: bool = True):
        """Set whether to use gradient checkpointing for memory efficiency."""
        self.text_model.config.use_gradient_checkpointing = use_gradient_checkpointing

        if self.audio_decoder is not None:
            self.audio_decoder.config.use_gradient_checkpointing = (
                use_gradient_checkpointing
            )


# ============================================================================
# Register models with AutoModel/AutoConfig for automatic loading
# ============================================================================

AutoConfig.register("fish_qwen3", FishQwen3Config)
AutoConfig.register("fish_qwen3_omni", FishQwen3OmniConfig)

AutoModel.register(FishQwen3Config, FishQwen3ForCausalLM)
AutoModel.register(FishQwen3OmniConfig, FishQwen3OmniForCausalLM)
