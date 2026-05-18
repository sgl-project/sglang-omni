# SPDX-License-Identifier: Apache-2.0
"""LLaDA2-Uni thinker: unified dLLM-MoE backbone."""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang_omni.models.weight_loader import default_weight_loader
from sglang_omni.vendor.sglang.core import ForwardBatch
from sglang_omni.vendor.sglang.distributed import get_tensor_model_parallel_world_size
from sglang_omni.vendor.sglang.layers import (
    AttentionType,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    QuantizationConfig,
    RadixAttention,
    RMSNorm,
    RowParallelLinear,
    SiluAndMul,
    StandardTopKOutput,
    VocabParallelEmbedding,
    get_moe_impl_class,
    get_rope,
)
from sglang_omni.vendor.sglang.models import (
    apply_qk_norm,
    create_fused_set_kv_buffer_arg,
    enable_fused_set_kv_buffer,
)
from sglang_omni.vendor.sglang.utils import make_layers


class LLaDA2MoeAttention(nn.Module):
    """Multi-head attention with GQA, partial RoPE, and QK normalization."""

    def __init__(
        self,
        config: PretrainedConfig,
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
        self.use_qk_norm = config.use_qk_norm

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_tp = self.num_heads // tp_size
        self.num_kv_heads_per_tp = max(1, self.num_kv_heads // tp_size)
        self.q_size = self.num_heads_per_tp * self.head_dim
        self.kv_size = self.num_kv_heads_per_tp * self.head_dim

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=config.use_qkv_bias,
            quant_config=quant_config,
        )
        self.dense = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )

        if hasattr(config, "partial_rotary_factor"):
            self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        elif hasattr(config, "rotary_dim"):
            self.rotary_dim = config.rotary_dim
        else:
            self.rotary_dim = self.head_dim

        # QK normalization layers (per-head)
        if self.use_qk_norm:
            self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # RoPE - using partial rotary factor
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        # Radix attention for paged KV cache
        self.attn = RadixAttention(
            self.num_heads_per_tp,
            self.head_dim,
            1.0 / math.sqrt(self.head_dim),
            self.num_kv_heads_per_tp,
            layer_id=layer_id,
            attn_type=AttentionType.ENCODER_ONLY,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # QK normalization (apply_qk_norm handles internal reshape to per-head)
        if self.use_qk_norm:
            q, k = apply_qk_norm(
                q, k, self.query_layernorm, self.key_layernorm, self.head_dim
            )

        # RoPE — sglang's rotary_emb handles partial rotation internally
        # via cos_sin_cache whose width equals rotary_dim (< head_dim).
        # Only the first rotary_dim dims are rotated; the rest pass through.
        q, k = self.rotary_emb(
            forward_batch.positions,
            q,
            k,
            fused_set_kv_buffer_arg=(
                create_fused_set_kv_buffer_arg(
                    value=v,
                    layer=self.attn,
                    forward_batch=forward_batch,
                )
                if enable_fused_set_kv_buffer(forward_batch)
                else None
            ),
        )

        attn_output = self.attn(
            q,
            k,
            v,
            forward_batch,
            save_kv_cache=not enable_fused_set_kv_buffer(forward_batch),
        )
        output, _ = self.dense(attn_output)
        return output


class LLaDA2MoeMLP(nn.Module):
    """Standard SiLU-gated MLP."""

    def __init__(
        self,
        config: PretrainedConfig,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class LLaDA2MoeGate(nn.Module):
    def __init__(
        self,
        config,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.weight = nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                dtype=self.params_dtype,
            ),
        )
        if getattr(config, "moe_router_enable_expert_bias", False):
            self.expert_bias = nn.Parameter(
                torch.empty((config.num_experts,), dtype=torch.float32),
            )
        else:
            self.expert_bias = None

    def forward(self, hidden_states):
        logits = F.linear(hidden_states.to(self.weight.dtype), self.weight, None).to(
            hidden_states.dtype
        )
        return logits


class LLaDA2MoeSparseMoeBlock(nn.Module):
    """Sparse MoE with group-limited top-k routing and optional shared expert."""

    def __init__(
        self,
        config: PretrainedConfig,
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

        # Gate always runs at half / full precision for now.
        router_dtype = getattr(config, "router_dtype", None)
        if router_dtype is None:
            self.router_dtype = None
        elif router_dtype == "fp32":
            self.router_dtype = torch.float32
        else:
            self.router_dtype = torch.bfloat16

        self.gate = LLaDA2MoeGate(
            config=config,
            params_dtype=self.router_dtype,
        )

        # FusedMoE implementation
        FusedMoE = get_moe_impl_class(quant_config)
        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            reduce_results=False,
        )

        # Shared expert
        if config.num_shared_experts and config.num_shared_experts > 0:
            shared_intermediate = (
                config.moe_intermediate_size * config.num_shared_experts
            )
            self.shared_experts = LLaDA2MoeMLP(
                config, shared_intermediate, quant_config
            )
        else:
            self.shared_experts = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Clone for shared expert input: FusedMoE with inplace=True overwrites
        # hidden_states, so the shared expert must use a separate copy.
        identity = (
            hidden_states.clone() if self.shared_experts is not None else hidden_states
        )

        # Router scores via sigmoid (not softmax like standard MoE)
        router_logits = self.gate(hidden_states)
        router_logits = router_logits.float()
        scores = torch.sigmoid(router_logits)

        # Add expert bias for load balancing
        if self.gate.expert_bias is not None:
            scores_for_routing = scores + self.gate.expert_bias
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

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
        )
        y = self.experts(hidden_states, topk_output)

        # Add shared expert output
        if self.shared_experts is not None:
            y = y + self.shared_experts(identity)

        return y

    def _group_limited_topk(
        self, scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Group-limited top-k expert selection."""
        num_tokens = scores.shape[0]
        experts_per_group = self.num_experts // self.n_group

        # Group scores: sum of top-2 experts per group
        group_scores = (
            scores.view(num_tokens, self.n_group, experts_per_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )

        # Select top groups
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
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


class LLaDA2MoeBlock(nn.Module):
    """Single transformer decoder layer with attention + MoE/MLP."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.is_dense = layer_id < config.first_k_dense_replace

        # Attention
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = LLaDA2MoeAttention(config, layer_id, quant_config)

        # FFN: dense MLP for first_k_dense_replace layers, MoE for rest
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if self.is_dense:
            self.mlp = LLaDA2MoeMLP(config, config.intermediate_size, quant_config)
        else:
            self.mlp = LLaDA2MoeSparseMoeBlock(config, layer_id, quant_config)

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

        hidden_states = self.attention(hidden_states, forward_batch)

        # FFN with post-attention norm
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class LLaDA2MoeTextModel(nn.Module):
    """text model body (no LM head)."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.word_embeddings = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix="": LLaDA2MoeBlock(config, idx, quant_config),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if input_embeds is not None:
            hidden_states = input_embeds
        else:
            hidden_states = self.word_embeddings(input_ids)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, forward_batch, residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights with prefix-based selection and MoE mapping."""

        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            prefix = "model."
            if name.startswith(prefix):
                name = name[len(prefix) :]

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    if name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)


class LLaDA2MoeModelLM(nn.Module):
    """Wraps LLaDA2MoeTextModel with LM head and LogitsProcessor.

    SGLang runtime expects:
      - model.model.word_embeddings  (embedding table)
      - model.model(...)          (text body forward)
      - model.lm_head             (output projection)
      - model.logits_processor    (logits post-processing)
    """

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config

        # Build model body
        self.model = LLaDA2MoeTextModel(config, quant_config)

        # Build LM head
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
        )

        # Build logits processor
        from sglang.srt.layers.logits_processor import LogitsProcessor

        self.logits_processor = LogitsProcessor(config, return_full_logits=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights from LLaDA2-Uni checkpoint.

        Routes lm_head weights to self.lm_head and text model weights
        to self.model.
        """
        model_weights = []
        lm_head_params = dict(self.lm_head.named_parameters())

        for name, tensor in weights:
            # Route lm_head weights
            if name == "lm_head.weight":
                param = lm_head_params.get("weight")
                if param is not None:
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, tensor)
                continue

            model_weights.append((name, tensor))

        self.model.load_weights(iter(model_weights))
