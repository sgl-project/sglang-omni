"""Qwen attention."""

import copy

import torch
import torch._dynamo
from torch import nn
from transformers import Qwen3Config
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.utils.deprecation import deprecate_kwarg

from sglang_omni.models.sensenova_u1.neo_unify.attention_backend import (
    position_ids_from_indexes,
)
from sglang_omni.models.sensenova_u1.neo_unify.attention_ops import (
    eager_attention_forward,
)
from sglang_omni.models.sensenova_u1.neo_unify.generation_attention import (
    forward_gen as generation_attention,
)
from sglang_omni.models.sensenova_u1.neo_unify.qwen_feed_forward import (
    SenseNovaRMSNorm,
    make_qwen3_rms_norm,
)
from sglang_omni.models.sensenova_u1.neo_unify.rotary_embeddings import (
    PositionEmbeddings,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

# Modified for SGLang; see this directory's README.md for upstream source.


__all__ = ["Qwen3Attention"]


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.config: Qwen3Config = config
        self.layer_idx: int = layer_idx
        self.head_dim: int = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups: int = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling: float = self.head_dim**-0.5
        self.attention_dropout: float = config.attention_dropout
        self.is_causal: bool = True

        self.q_proj: nn.Linear = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.q_proj_mot_gen: nn.Linear = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )

        self.k_proj: nn.Linear = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj_mot_gen: nn.Linear = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )

        self.v_proj: nn.Linear = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj_mot_gen: nn.Linear = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj: nn.Linear = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.o_proj_mot_gen: nn.Linear = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.q_norm: SenseNovaRMSNorm = make_qwen3_rms_norm(
            self.head_dim // 2, eps=config.rms_norm_eps
        )
        self.q_norm_mot_gen: SenseNovaRMSNorm = make_qwen3_rms_norm(
            self.head_dim // 2, eps=config.rms_norm_eps
        )
        self.q_norm_hw: SenseNovaRMSNorm = make_qwen3_rms_norm(
            self.head_dim // 2, eps=config.rms_norm_eps
        )
        self.q_norm_hw_mot_gen: SenseNovaRMSNorm = make_qwen3_rms_norm(
            self.head_dim // 2, eps=config.rms_norm_eps
        )

        self.k_norm: SenseNovaRMSNorm = make_qwen3_rms_norm(
            self.head_dim // 2, eps=config.rms_norm_eps
        )
        self.k_norm_mot_gen: SenseNovaRMSNorm = make_qwen3_rms_norm(
            self.head_dim // 2, eps=config.rms_norm_eps
        )
        self.k_norm_hw: SenseNovaRMSNorm = make_qwen3_rms_norm(
            self.head_dim // 2, eps=config.rms_norm_eps
        )
        self.k_norm_hw_mot_gen: SenseNovaRMSNorm = make_qwen3_rms_norm(
            self.head_dim // 2, eps=config.rms_norm_eps
        )

        self.sliding_window: int | None = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

        t_config = copy.deepcopy(config)
        t_config.head_dim = config.head_dim // 2
        self.rotary_emb: Qwen3RotaryEmbedding = Qwen3RotaryEmbedding(config=t_config)

        hw_config = copy.deepcopy(config)
        hw_config.head_dim = config.head_dim // 4
        hw_config.rope_theta = config.rope_theta_hw
        if isinstance(getattr(hw_config, "rope_parameters", None), dict):
            hw_config.rope_parameters = {
                **hw_config.rope_parameters,
                "rope_theta": config.rope_theta_hw,
            }
        else:
            pass
        hw_config.max_position_embeddings = config.max_position_embeddings_hw
        self.rotary_emb_hw: Qwen3RotaryEmbedding = Qwen3RotaryEmbedding(
            config=hw_config
        )

    def resolve_rope_tables(
        self,
        indexes: torch.LongTensor,
        hidden_states: torch.Tensor,
        position_embeddings: PositionEmbeddings | None = None,
    ) -> PositionEmbeddings:
        if (
            position_embeddings is None
            or position_embeddings[0][0].dtype != hidden_states.dtype
        ):
            position_embeddings = (
                self.rotary_emb(hidden_states, position_ids_from_indexes(indexes, 0)),
                self.rotary_emb_hw(
                    hidden_states, position_ids_from_indexes(indexes, 1)
                ),
                self.rotary_emb_hw(
                    hidden_states, position_ids_from_indexes(indexes, 2)
                ),
            )
        else:
            pass
        return position_embeddings

    def forward_und(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.LongTensor | None,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: PositionEmbeddings | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert (
            self.config._attn_implementation
            == "eager"  # noqa: leading-underscore -- Transformers attention backend field
        )
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states_t, query_states_hw = query_states.chunk(2, dim=-1)
        query_states_t = self.q_norm(query_states_t).transpose(1, 2)
        query_states_hw = self.q_norm_hw(query_states_hw).transpose(1, 2)
        query_states_h, query_states_w = query_states_hw.chunk(2, dim=-1)

        key_states = self.k_proj(hidden_states).view(hidden_shape)
        key_states_t, key_states_hw = key_states.chunk(2, dim=-1)
        key_states_t = self.k_norm(key_states_t).transpose(1, 2)
        key_states_hw = self.k_norm_hw(key_states_hw).transpose(1, 2)
        key_states_h, key_states_w = key_states_hw.chunk(2, dim=-1)

        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        (cos_t, sin_t), (cos_h, sin_h), (cos_w, sin_w) = self.resolve_rope_tables(
            indexes, hidden_states, position_embeddings
        )
        query_states_t, key_states_t = apply_rotary_pos_emb(
            query_states_t, key_states_t, cos_t, sin_t
        )
        query_states_h, key_states_h = apply_rotary_pos_emb(
            query_states_h, key_states_h, cos_h, sin_h
        )
        query_states_w, key_states_w = apply_rotary_pos_emb(
            query_states_w, key_states_w, cos_w, sin_w
        )

        query_states = torch.cat(
            [query_states_t, query_states_h, query_states_w], dim=-1
        )
        key_states = torch.cat([key_states_t, key_states_h, key_states_w], dim=-1)

        if past_key_values is not None:
            update_cache = kwargs.get("update_cache", True)
            if update_cache:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs=None
                )
            else:
                layer = past_key_values.layers[self.layer_idx]
                past_k, past_v = layer.keys, layer.values

                if past_k is not None:
                    key_states = torch.cat([past_k, key_states], dim=2)
                    value_states = torch.cat([past_v, value_states], dim=2)
                else:
                    pass
        else:
            pass

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    forward_gen = generation_attention

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: torch.LongTensor | None,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: PositionEmbeddings | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if exist_non_image_gen_tokens and not exist_image_gen_tokens:
            return self.forward_und(
                hidden_states,
                indexes,
                attention_mask,
                past_key_values,
                cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        else:
            pass
        if not exist_non_image_gen_tokens and exist_image_gen_tokens:
            return self.forward_gen(
                hidden_states,
                indexes,
                attention_mask,
                past_key_values,
                cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        else:
            pass

        raise NotImplementedError(
            "The mixed und/gen forward path is not yet validated (issue #207): known "
            "issues are fixed, but it has no parity test and no production caller. "
            "Split the sequence at token-type boundaries and use forward_und / forward_gen."
        )
