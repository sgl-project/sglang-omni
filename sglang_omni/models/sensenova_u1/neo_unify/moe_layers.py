"""SenseNova expert routing and decoder layers."""

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.deprecation import deprecate_kwarg

from sglang_omni.models.sensenova_u1.neo_unify.configuration_neo_chat import (
    NEOMoELLMConfig,
)
from sglang_omni.models.sensenova_u1.neo_unify.qwen_attention import Qwen3Attention
from sglang_omni.models.sensenova_u1.neo_unify.qwen_feed_forward import (
    SenseNovaRMSNorm,
    make_qwen3_rms_norm,
)

# Modified for SGLang; see this directory's README.md for upstream source.


class Qwen3MoeMLP(nn.Module):
    """Feed-forward network for one routed expert."""

    def __init__(
        self, config: NEOMoELLMConfig, intermediate_size: int | None = None
    ) -> None:
        super().__init__()

        self.config: NEOMoELLMConfig = config
        self.hidden_size: int = config.hidden_size
        self.intermediate_size: int = (
            intermediate_size
            if intermediate_size is not None
            else config.intermediate_size
        )
        self.gate_proj: nn.Linear = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj: nn.Linear = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj: nn.Linear = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False
        )
        self.act_fn: nn.Module = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3MoeSparseMoeBlock(nn.Module):
    """Top-k softmax-routed MoE block matching HuggingFace's Qwen3-MoE layout."""

    def __init__(
        self,
        config: NEOMoELLMConfig,
        num_experts: int | None = None,
        num_experts_per_tok: int | None = None,
        moe_intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        self.num_experts: int = (
            int(num_experts) if num_experts is not None else int(config.num_experts)
        )
        self.top_k: int = int(
            num_experts_per_tok
            if num_experts_per_tok is not None
            else config.num_experts_per_tok
        )
        self.norm_topk_prob: bool = bool(getattr(config, "norm_topk_prob", True))
        self.hidden_size: int = config.hidden_size

        expert_intermediate_size = int(
            moe_intermediate_size
            if moe_intermediate_size is not None
            else config.moe_intermediate_size
        )

        self.gate: nn.Linear = nn.Linear(
            config.hidden_size, self.num_experts, bias=False
        )
        self.experts: nn.ModuleList = nn.ModuleList(
            [
                Qwen3MoeMLP(config, intermediate_size=expert_intermediate_size)
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = orig_shape[-1]
        flat = hidden_states.view(-1, hidden_dim)
        n_tokens = flat.shape[0]

        router_logits = self.gate(flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True
            )
        else:
            pass
        routing_weights = routing_weights.to(flat.dtype)

        output = torch.zeros(
            (n_tokens, hidden_dim), dtype=flat.dtype, device=flat.device
        )
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(
            2, 1, 0
        )

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            else:
                pass
            expert_layer = self.experts[expert_idx]
            current_state = flat.index_select(0, top_x)
            current_out = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )
            output.index_add_(0, top_x, current_out.to(flat.dtype))

        return output.view(*orig_shape)


class Qwen3MoeDecoderLayer(GradientCheckpointingLayer):
    """A Qwen3-MoE decoder block with the NEO-Unify two-branch structure."""

    def __init__(self, config: NEOMoELLMConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size: int = config.hidden_size
        self.self_attn: Qwen3Attention = Qwen3Attention(
            config=config, layer_idx=layer_idx
        )

        mlp_only_layers = list(getattr(config, "mlp_only_layers", []) or [])
        decoder_sparse_step = int(getattr(config, "decoder_sparse_step", 1) or 1)
        is_sparse = (
            int(config.num_experts) > 0
            and layer_idx not in mlp_only_layers
            and (layer_idx + 1) % decoder_sparse_step == 0
        )

        if is_sparse:
            self.mlp: Qwen3MoeSparseMoeBlock = Qwen3MoeSparseMoeBlock(
                config,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                moe_intermediate_size=config.moe_intermediate_size,
            )
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)

        self.mlp_mot_gen: Qwen3MoeSparseMoeBlock = Qwen3MoeSparseMoeBlock(
            config,
            num_experts=getattr(config, "gen_num_experts", config.num_experts),
            num_experts_per_tok=getattr(
                config, "gen_num_experts_per_tok", config.num_experts_per_tok
            ),
            moe_intermediate_size=getattr(
                config, "gen_moe_intermediate_size", config.moe_intermediate_size
            ),
        )

        self.input_layernorm: SenseNovaRMSNorm = make_qwen3_rms_norm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.input_layernorm_mot_gen: SenseNovaRMSNorm = make_qwen3_rms_norm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm: SenseNovaRMSNorm = make_qwen3_rms_norm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm_mot_gen: SenseNovaRMSNorm = make_qwen3_rms_norm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention_type: str = config.layer_types[layer_idx]

    def forward_und(
        self,
        hidden_states: torch.Tensor,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            image_gen_indicators=image_gen_indicators,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def forward_gen(
        self,
        hidden_states: torch.Tensor,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm_mot_gen(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            image_gen_indicators=image_gen_indicators,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm_mot_gen(hidden_states)
        hidden_states = self.mlp_mot_gen(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        if exist_non_image_gen_tokens and not exist_image_gen_tokens:
            return self.forward_und(
                hidden_states,
                image_gen_indicators,
                exist_non_image_gen_tokens,
                exist_image_gen_tokens,
                indexes,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
                cache_position,
                **kwargs,
            )
        else:
            pass
        if not exist_non_image_gen_tokens and exist_image_gen_tokens:
            return self.forward_gen(
                hidden_states,
                image_gen_indicators,
                exist_non_image_gen_tokens,
                exist_image_gen_tokens,
                indexes,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
                cache_position,
                **kwargs,
            )
        else:
            pass

        raise NotImplementedError(
            "Mixed und/gen decoder-layer forward is not yet validated (issue #207). "
            "Split the sequence at token-type boundaries and use forward_und / forward_gen."
        )
