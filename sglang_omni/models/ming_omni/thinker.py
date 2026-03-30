# SPDX-License-Identifier: Apache-2.0
"""SGLang model class for BailingMoeV2 (Ming-Omni thinker).

This implements the BailingMoeV2ForCausalLM as a native SGLang model,
enabling paged KV cache, RadixAttention, and FusedMoE support.

Architecture (from config.json):
  - 32 layers, 32 attention heads, 4 KV heads (GQA)
  - Hidden 4096, intermediate 9216, MoE intermediate 1024
  - 256 experts, 8/token, 1 shared expert, MultiRouter
  - partial_rotary_factor=0.5, rope_theta=2.4M
  - use_qk_norm=True, use_expert_bias=True
  - first_k_dense_replace=1 (layer 0 is dense, rest are MoE)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang_omni.models.weight_loader import default_weight_loader
from sglang_omni.utils import add_prefix
from sglang_omni.vendor.sglang.core import ForwardBatch
from sglang_omni.vendor.sglang.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang_omni.vendor.sglang.layers import (
    LayerCommunicator,
    LayerScatterModes,
    QKVParallelLinear,
    QuantizationConfig,
    RadixAttention,
    ReplicatedLinear,
    RMSNorm,
    RoutingMethodType,
    RowParallelLinear,
    TopK,
    VocabParallelEmbedding,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_moe_impl_class,
    get_rope,
)
from sglang_omni.vendor.sglang.models import (
    apply_qk_norm,
    create_fused_set_kv_buffer_arg,
    enable_fused_set_kv_buffer,
)
from sglang_omni.vendor.sglang.server_args import get_global_server_args
from sglang_omni.vendor.sglang.utils import make_layers

logger = logging.getLogger(__name__)


class BailingMoeV2Config(PretrainedConfig):
    """Adapter config that maps Ming's config.json to SGLang expectations."""

    model_type = "bailing_moe_v2"

    def __init__(
        self,
        vocab_size=157184,
        hidden_size=4096,
        intermediate_size=9216,
        moe_intermediate_size=1024,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        max_position_embeddings=32768,
        rope_theta=2400000.0,
        partial_rotary_factor=0.5,
        use_qk_norm=True,
        use_qkv_bias=False,
        num_experts=256,
        num_experts_per_tok=8,
        num_shared_experts=1,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        use_expert_bias=True,
        first_k_dense_replace=1,
        rope_scaling=None,
        tie_word_embeddings=False,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.use_qk_norm = use_qk_norm
        self.use_qkv_bias = use_qkv_bias
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.use_expert_bias = use_expert_bias
        self.first_k_dense_replace = first_k_dense_replace
        self.rope_scaling = rope_scaling

        # Derived
        self.head_dim = hidden_size // num_attention_heads
        self.rotary_dim = int(self.head_dim * partial_rotary_factor)


# ============================================================================
# Attention Layer
# ============================================================================

class BailingMoeV2Attention(nn.Module):
    """Multi-head attention with GQA, partial RoPE, and QK normalization."""

    def __init__(
        self,
        config: BailingMoeV2Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.use_qk_norm = config.use_qk_norm

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_tp = self.num_heads // tp_size
        self.num_kv_heads_per_tp = max(1, self.num_kv_heads // tp_size)
        self.q_size = self.num_heads_per_tp * self.head_dim
        self.kv_size = self.num_kv_heads_per_tp * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=config.use_qkv_bias,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )

        # QK normalization layers (per-head)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # RoPE - using partial rotary factor
        self.rotary_emb = get_rope(
            self.rotary_dim,
            rotary_dim=self.rotary_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        # Radix attention for paged KV cache
        self.attn = RadixAttention(
            self.num_heads_per_tp,
            self.head_dim,
            1.0 / math.sqrt(self.head_dim),
            self.num_kv_heads_per_tp,
            layer_id=layer_id,
        )

    def forward_prepare(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """QKV projection + QK norm + RoPE."""
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Reshape for multi-head
        q = q.view(-1, self.num_heads_per_tp, self.head_dim)
        k = k.view(-1, self.num_kv_heads_per_tp, self.head_dim)
        v = v.view(-1, self.num_kv_heads_per_tp, self.head_dim)

        # QK normalization
        if self.use_qk_norm:
            q = apply_qk_norm(q, self.q_norm)
            k = apply_qk_norm(k, self.k_norm)

        # Partial RoPE: only apply to first rotary_dim dimensions
        q_rot = q[..., : self.rotary_dim]
        q_pass = q[..., self.rotary_dim :]
        k_rot = k[..., : self.rotary_dim]
        k_pass = k[..., self.rotary_dim :]

        q_rot, k_rot = self.rotary_emb(
            forward_batch.positions, q_rot, k_rot
        )

        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)

        return q, k, v

    def forward_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Attention computation with paged KV cache."""
        attn_output = self.attn(q, k, v, forward_batch)
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        q, k, v = self.forward_prepare(hidden_states, forward_batch)
        attn_output = self.forward_core(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


# ============================================================================
# MLP (for dense layers and shared experts)
# ============================================================================

class BailingMoeV2MLP(nn.Module):
    """Standard SiLU-gated MLP."""

    def __init__(
        self,
        config: BailingMoeV2Config,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.gate_up_proj = ReplicatedLinear(
            config.hidden_size,
            intermediate_size * 2,
            bias=False,
            quant_config=quant_config,
        )
        self.down_proj = ReplicatedLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(torch.nn.functional.silu(gate) * up)[0]


# ============================================================================
# Sparse MoE Block
# ============================================================================

class BailingMoeV2SparseMoeBlock(nn.Module):
    """Sparse MoE with group-limited top-k routing and optional shared expert.

    Routing: group_limited_topk
      1. Divide 256 experts into n_group=8 groups of 32
      2. Score each group by sum of top-2 expert scores within group
      3. Select topk_group=4 groups
      4. From selected groups, pick top num_experts_per_tok=8 experts
      5. Apply sigmoid gating with routed_scaling_factor=2.5
    """

    def __init__(
        self,
        config: BailingMoeV2Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor

        # Gate: linear projection for router scores
        self.gate = ReplicatedLinear(
            config.hidden_size, config.num_experts, bias=False
        )

        # Expert bias for load balancing
        if config.use_expert_bias:
            self.expert_bias = nn.Parameter(
                torch.zeros(config.num_experts), requires_grad=False
            )
        else:
            self.expert_bias = None

        # FusedMoE implementation
        FusedMoE = get_moe_impl_class()
        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            tp_size=get_tensor_model_parallel_world_size(),
        )

        # Shared expert
        if config.num_shared_experts and config.num_shared_experts > 0:
            shared_intermediate = (
                config.moe_intermediate_size * config.num_shared_experts
            )
            self.shared_experts = BailingMoeV2MLP(
                config, shared_intermediate, quant_config
            )
        else:
            self.shared_experts = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states

        # Router scores via sigmoid (not softmax like standard MoE)
        router_logits, _ = self.gate(hidden_states)
        router_logits = router_logits.float()
        scores = torch.sigmoid(router_logits)

        # Add expert bias for load balancing
        if self.expert_bias is not None:
            scores_for_routing = scores + self.expert_bias
        else:
            scores_for_routing = scores

        # Group-limited top-k selection
        topk_weights, topk_ids = self._group_limited_topk(scores_for_routing)

        # Gather actual scores (without bias) for the selected experts
        topk_weights = torch.gather(scores, dim=1, index=topk_ids)

        # Normalize and scale
        if self.num_experts_per_tok > 1:
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        topk_weights = topk_weights * self.routed_scaling_factor

        # FusedMoE forward
        y = self.experts(hidden_states, topk_ids, topk_weights)

        # Add shared expert output
        if self.shared_experts is not None:
            y = y + self.shared_experts(identity)

        return y

    def _group_limited_topk(
        self, scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Group-limited top-k expert selection.

        1. Reshape scores to [tokens, n_group, experts_per_group]
        2. Score each group by sum of top-2 within group
        3. Select top topk_group groups
        4. Mask non-selected groups to -inf
        5. Select top num_experts_per_tok from unmasked experts
        """
        num_tokens = scores.shape[0]
        experts_per_group = self.num_experts // self.n_group

        # Group scores: sum of top-2 experts per group
        group_scores = (
            scores.view(num_tokens, self.n_group, experts_per_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )

        # Select top groups
        group_idx = torch.topk(
            group_scores, k=self.topk_group, dim=-1, sorted=False
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        # Expand group mask to expert-level
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, experts_per_group)
            .reshape(num_tokens, -1)
        )

        # Mask and select top-k
        masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
        topk_weights, topk_ids = torch.topk(
            masked_scores, k=self.num_experts_per_tok, dim=-1, sorted=False
        )

        return topk_weights, topk_ids


# ============================================================================
# Decoder Layer
# ============================================================================

class BailingMoeV2DecoderLayer(nn.Module):
    """Single transformer decoder layer with attention + MoE/MLP."""

    def __init__(
        self,
        config: BailingMoeV2Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.is_dense = layer_id < config.first_k_dense_replace

        # Attention
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = BailingMoeV2Attention(config, layer_id, quant_config)

        # FFN: dense MLP for first_k_dense_replace layers, MoE for rest
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if self.is_dense:
            self.mlp = BailingMoeV2MLP(
                config, config.intermediate_size, quant_config
            )
        else:
            self.mlp = BailingMoeV2SparseMoeBlock(config, layer_id, quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self attention with pre-norm
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(hidden_states, forward_batch)

        # FFN with post-attention norm
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# ============================================================================
# Full Model
# ============================================================================

class BailingMoeV2TextModel(nn.Module):
    """BailingMoeV2 text model body (no LM head)."""

    def __init__(
        self,
        config: BailingMoeV2Config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda layer_id: BailingMoeV2DecoderLayer(
                config, layer_id, quant_config
            ),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is not None:
            hidden_states = input_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, forward_batch, residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights with prefix-based selection and MoE mapping."""
        from sglang_omni.vendor.sglang.models import (
            maybe_update_fused_qkv_proj,
            maybe_update_fused_moe_proj,
            extract_fused_experts,
        )

        stacked_params_mapping = [
            # SGLang fused QKV
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # SGLang fused gate+up
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Map Ming weight names to SGLang names
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            # Strip common prefixes from Ming checkpoint
            for prefix in ("model.model.", "model.", "thinker.model.", "thinker."):
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break

            # Handle MoE expert weights
            # Ming format: layers.X.mlp.experts.Y.gate_proj.weight
            # SGLang FusedMoE: layers.X.mlp.experts.w1 (gate), w2 (down), w3 (up)
            if ".experts." in name and ".mlp." in name:
                # Try to map to fused expert format
                loaded = extract_fused_experts(
                    name, loaded_weight, params_dict
                )
                if loaded:
                    continue

            # Handle fused QKV
            handled = maybe_update_fused_qkv_proj(
                name,
                loaded_weight,
                stacked_params_mapping,
                params_dict,
            )
            if handled:
                continue

            # Handle fused gate+up for shared experts
            handled = maybe_update_fused_moe_proj(
                name,
                loaded_weight,
                stacked_params_mapping,
                params_dict,
            )
            if handled:
                continue

            # Direct parameter mapping
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                logger.debug("Skipping weight: %s", name)
