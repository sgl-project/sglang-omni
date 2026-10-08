# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni vision attention with platform rotary and shared patch counts."""

from __future__ import annotations

import torch
from torch import nn
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoderConfig,
)
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs

from sglang_omni.models.qwen3_omni.components.vision_compat import (
    VisionRotaryInputs,
    VisionSequenceMetadata,
)
from sglang_omni.platforms.interface import JointRopeInplaceKernel


class Qwen3OmniVisionAttention(nn.Module):
    def __init__(
        self,
        attention: hf_modeling.Qwen3OmniMoeVisionAttention,
        *,
        joint_rope_kernel: JointRopeInplaceKernel | None,
    ) -> None:
        super().__init__()
        self.qkv: nn.Linear = attention.qkv
        self.proj: nn.Linear = attention.proj
        self.config: Qwen3OmniMoeVisionEncoderConfig = attention.config
        self.num_heads: int = attention.num_heads
        self.num_key_value_groups: int = attention.num_key_value_groups
        self.scaling: float = attention.scaling
        self.attention_dropout: float = attention.attention_dropout
        self.is_causal: bool = attention.is_causal
        self.joint_rope_kernel: JointRopeInplaceKernel | None = joint_rope_kernel
        self.train(attention.training)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | VisionRotaryInputs,
        *,
        sequence_metadata: VisionSequenceMetadata,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        sequence_patch_count = hidden_states.shape[0]
        query, key, value = (
            self.qkv(hidden_states)
            .reshape(sequence_patch_count, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        if self.joint_rope_kernel is None:
            cos, sin = position_embeddings
            query, key = hf_modeling.apply_rotary_pos_emb_vision(query, key, cos, sin)
        else:
            self.joint_rope_kernel(
                query,
                key,
                position_embeddings.cos_sin_cache,
                position_embeddings.positions,
                is_neox=True,
            )

        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)
        attention_interface = hf_modeling.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,  # noqa: leading-underscore  # Transformers configuration field
            hf_modeling.eager_attention_forward,
        )
        if hf_modeling.is_flash_attention_requested(self.config):
            attention_output, _ = attention_interface(
                self,
                query,
                key,
                value,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=sequence_metadata.max_sequence_patch_count,
                max_length_k=sequence_metadata.max_sequence_patch_count,
                is_causal=False,
                **kwargs,
            )
        else:
            query_segments, key_segments, value_segments = (
                torch.split(tensor, sequence_metadata.sequence_patch_counts, dim=2)
                for tensor in (query, key, value)
            )
            attention_outputs = [
                attention_interface(
                    self,
                    query_segment,
                    key_segment,
                    value_segment,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for query_segment, key_segment, value_segment in zip(
                    query_segments, key_segments, value_segments
                )
            ]
            attention_output = torch.cat(attention_outputs, dim=1)
        attention_output = attention_output.reshape(
            sequence_patch_count, -1
        ).contiguous()
        attention_output = self.proj(attention_output)
        return attention_output
