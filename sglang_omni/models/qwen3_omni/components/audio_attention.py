# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# Attention forwarding is adapted from Transformers' Qwen3-Omni audio attention.
"""Loaded Qwen3-Omni audio attention with a single QKV projection."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling
from transformers.processing_utils import Unpack
from transformers.utils.generic import is_flash_attention_requested


class SegmentSplits:
    """Per-request attention segment sizes, shared by every encoder layer."""

    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value: list[int] | None = None


class FusedAudioAttention(nn.Module):
    """Pack an already-loaded attention module without retaining duplicate weights."""

    def __init__(
        self,
        attention: hf_modeling.Qwen3OmniMoeAudioAttention,
        splits: SegmentSplits,
    ) -> None:
        super().__init__()
        self.config = attention.config
        self.embed_dim: int = attention.embed_dim
        self.num_heads: int = attention.num_heads
        self.head_dim: int = attention.head_dim
        self.num_key_value_groups: int = attention.num_key_value_groups
        self.scaling: float = attention.scaling
        self.attention_dropout: float = attention.attention_dropout
        self.is_causal: bool = attention.is_causal
        self.is_decoder: bool = attention.is_decoder
        self.splits = splits
        self.out_proj = attention.out_proj
        projections = (attention.q_proj, attention.k_proj, attention.v_proj)
        has_bias = any(projection.bias is not None for projection in projections)
        self.qkv_proj = nn.Linear(
            self.embed_dim, 3 * self.embed_dim, bias=has_bias, device="meta"
        )
        self.qkv_proj.weight = nn.Parameter(
            torch.cat([projection.weight.detach() for projection in projections]),
            requires_grad=any(
                projection.weight.requires_grad for projection in projections
            ),
        )
        if has_bias:
            self.qkv_proj.bias = nn.Parameter(
                torch.cat(
                    [
                        (
                            projection.bias.detach()
                            if projection.bias is not None
                            else projection.weight.new_zeros(self.embed_dim)
                        )
                        for projection in projections
                    ]
                ),
                requires_grad=any(
                    projection.bias is not None and projection.bias.requires_grad
                    for projection in projections
                ),
            )
        else:
            pass
        self.train(attention.training)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        sequence_length = hidden_states.shape[0]
        query, key, value = (
            states.transpose(0, 1).unsqueeze(0)
            for states in project_audio_qkv(self, hidden_states)
        )
        attention_interface = hf_modeling.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,  # noqa: leading-underscore  # Transformers configuration field.
            hf_modeling.eager_attention_forward,
        )
        splits = self.splits.value
        use_shared_splits = splits is not None and sum(splits) == sequence_length
        if not use_shared_splits and is_flash_attention_requested(self.config):
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attention_output = attention_interface(
                self,
                query,
                key,
                value,
                attention_mask=None,
                scaling=self.scaling,
                dropout=self.attention_dropout if self.training else 0.0,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )[0]
        else:
            if not use_shared_splits:
                splits = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            else:
                pass
            attention_output = torch.cat(
                [
                    attention_interface(
                        self,
                        query_segment,
                        key_segment,
                        value_segment,
                        attention_mask=None,
                        scaling=self.scaling,
                        dropout=self.attention_dropout if self.training else 0.0,
                        is_causal=False,
                        **kwargs,
                    )[0]
                    for query_segment, key_segment, value_segment in zip(
                        *(
                            torch.split(states, splits, dim=2)
                            for states in (query, key, value)
                        )
                    )
                ],
                dim=1,
            )
        return self.out_proj(attention_output.reshape(sequence_length, -1).contiguous())


def project_audio_qkv(
    attention: hf_modeling.Qwen3OmniMoeAudioAttention | FusedAudioAttention,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project packed audio tokens into token, head, and channel views."""
    if isinstance(attention, FusedAudioAttention):
        query, key, value = attention.qkv_proj(hidden_states).chunk(3, dim=-1)
    else:
        query = attention.q_proj(hidden_states)
        key = attention.k_proj(hidden_states)
        value = attention.v_proj(hidden_states)
    shape = (hidden_states.shape[0], attention.num_heads, -1)
    return query.reshape(shape), key.reshape(shape), value.reshape(shape)
