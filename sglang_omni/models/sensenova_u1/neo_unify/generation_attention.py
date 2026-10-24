"""Attention operations shared by dense and MoE backbones."""

from typing import Protocol

import torch
import torch._dynamo
from torch import nn
from transformers import Qwen3Config
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack

from sglang_omni.models.sensenova_u1.neo_unify.attention_backend import flash_or_sdpa
from sglang_omni.models.sensenova_u1.neo_unify.attention_ops import (
    AttentionForward,
    AttentionModule,
    eager_attention_forward,
)
from sglang_omni.models.sensenova_u1.neo_unify.rotary_embeddings import (
    PositionEmbeddings,
    apply_rotary_pos_emb,
)

# Modified for SGLang; see this directory's README.md for upstream source.


class AttentionState(AttentionModule, Protocol):
    config: Qwen3Config
    head_dim: int
    layer_idx: int
    scaling: float
    sliding_window: int | None
    attention_dropout: float
    k_norm_hw_mot_gen: nn.Module
    k_norm_mot_gen: nn.Module
    k_proj_mot_gen: nn.Module
    o_proj_mot_gen: nn.Module
    q_norm_hw_mot_gen: nn.Module
    q_norm_mot_gen: nn.Module
    q_proj_mot_gen: nn.Module
    v_proj_mot_gen: nn.Module

    def resolve_rope_tables(
        self,
        indexes: torch.LongTensor,
        hidden_states: torch.Tensor,
        position_embeddings: PositionEmbeddings | None = None,
    ) -> PositionEmbeddings:
        """Return temporal and spatial rotary embeddings for the input positions."""


def forward_gen(
    self: AttentionState,
    hidden_states: torch.Tensor,
    indexes: torch.LongTensor | None,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    position_embeddings: PositionEmbeddings | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj_mot_gen(hidden_states).view(hidden_shape)
    query_states_t, query_states_hw = query_states.chunk(2, dim=-1)
    query_states_t = self.q_norm_mot_gen(query_states_t).transpose(1, 2)
    query_states_hw = self.q_norm_hw_mot_gen(query_states_hw).transpose(1, 2)
    query_states_h, query_states_w = query_states_hw.chunk(2, dim=-1)

    key_states = self.k_proj_mot_gen(hidden_states).view(hidden_shape)
    key_states_t, key_states_hw = key_states.chunk(2, dim=-1)
    key_states_t = self.k_norm_mot_gen(key_states_t).transpose(1, 2)
    key_states_hw = self.k_norm_hw_mot_gen(key_states_hw).transpose(1, 2)
    key_states_h, key_states_w = key_states_hw.chunk(2, dim=-1)

    value_states = self.v_proj_mot_gen(hidden_states).view(hidden_shape).transpose(1, 2)

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

    query_states = torch.cat([query_states_t, query_states_h, query_states_w], dim=-1)
    key_states = torch.cat([key_states_t, key_states_h, key_states_w], dim=-1)

    update_cache = kwargs.get("update_cache", True)

    key_padding_mask = (
        attention_mask is not None
        and attention_mask.ndim == 4
        and (attention_mask.shape[-2] == 1 or attention_mask.dtype == torch.bool)
    )
    if attention_mask is None or key_padding_mask:
        actual_seq_lengths_kv = None
        kv_padding_size = None
        input_layout = "BSND"
        layer = (
            past_key_values.layers[self.layer_idx]
            if past_key_values is not None and not update_cache
            else None
        )
        use_bnsd_cache = (
            layer is not None
            and getattr(layer, "flash_cache_layout", None) == "BNSD"
            and getattr(layer, "flash_k_cache", None) is not None
            and getattr(layer, "flash_v_cache", None) is not None
        )

        if use_bnsd_cache:
            q = query_states
            k_cur = key_states
            v_cur = value_states
            prefix_len = layer.flash_prefix_len
            cur_len = k_cur.shape[2]
            layer.flash_k_cache[:, :, prefix_len : prefix_len + cur_len].copy_(k_cur)
            layer.flash_v_cache[:, :, prefix_len : prefix_len + cur_len].copy_(v_cur)
            k = layer.flash_k_cache[:, :, : prefix_len + cur_len]
            v = layer.flash_v_cache[:, :, : prefix_len + cur_len]
            actual_seq_lengths_kv = layer.flash_actual_seq_lengths_kv
            kv_padding_size = layer.flash_kv_padding_size
            input_layout = "BNSD"
        else:
            q = query_states.transpose(1, 2).contiguous()
            k_cur = key_states.transpose(1, 2).contiguous()
            v_cur = value_states.transpose(1, 2).contiguous()

            if past_key_values is not None:
                if update_cache:
                    key_states, value_states = past_key_values.update(
                        key_states, value_states, self.layer_idx, cache_kwargs=None
                    )
                    k = key_states.transpose(1, 2).contiguous()
                    v = value_states.transpose(1, 2).contiguous()
                else:
                    if (
                        getattr(layer, "flash_k_cache", None) is not None
                        and getattr(layer, "flash_v_cache", None) is not None
                    ):
                        prefix_len = layer.flash_prefix_len
                        cur_len = k_cur.shape[1]
                        layer.flash_k_cache[:, prefix_len : prefix_len + cur_len].copy_(
                            k_cur
                        )
                        layer.flash_v_cache[:, prefix_len : prefix_len + cur_len].copy_(
                            v_cur
                        )
                        k = layer.flash_k_cache[:, : prefix_len + cur_len]
                        v = layer.flash_v_cache[:, : prefix_len + cur_len]
                        actual_seq_lengths_kv = getattr(
                            layer, "flash_actual_seq_lengths_kv", None
                        )
                        kv_padding_size = getattr(layer, "flash_kv_padding_size", None)
                        if attention_mask is None:
                            attention_mask = layer.flash_key_padding_mask
                        else:
                            pass
                    else:
                        past_k, past_v = layer.keys, layer.values
                        if past_k is not None:
                            past_k = past_k.transpose(1, 2).contiguous()
                            past_v = past_v.transpose(1, 2).contiguous()
                            k = torch.cat([past_k, k_cur], dim=1)
                            v = torch.cat([past_v, v_cur], dim=1)
                        else:
                            k = k_cur
                            v = v_cur
            else:
                k = k_cur
                v = v_cur

        assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
        assert q.shape[0] == k.shape[0] == v.shape[0], (q.shape, k.shape, v.shape)
        assert k.shape[1] == v.shape[1], (k.shape, v.shape)
        assert k.shape[2] == v.shape[2], (k.shape, v.shape)
        assert q.shape[3] == k.shape[3] == v.shape[3], (q.shape, k.shape, v.shape)

        attn_output = flash_or_sdpa(
            q,
            k,
            v,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            softmax_scale=self.scaling,
            causal=False,
            attention_mask=attention_mask,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            kv_padding_size=kv_padding_size,
            input_layout=input_layout,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj_mot_gen(attn_output)
        return attn_output, None
    else:
        pass

    if past_key_values is not None:
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

    attention_interface: AttentionForward = eager_attention_forward
    if (
        self.config._attn_implementation
        != "eager"  # noqa: leading-underscore -- Transformers attention backend field
    ):
        attention_interface = ALL_ATTENTION_FUNCTIONS[
            self.config._attn_implementation  # noqa: leading-underscore -- Transformers attention backend field
        ]
    else:
        pass

    attn_output, attn_weights = attention_interface(
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
    attn_output = self.o_proj_mot_gen(attn_output)
    return attn_output, attn_weights
