# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni vision encoder with shared rotary and attention inputs."""

from __future__ import annotations

import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoderConfig,
)

from sglang_omni.models.qwen3_omni.components.vision_attention import (
    Qwen3OmniVisionAttention,
)
from sglang_omni.models.qwen3_omni.components.vision_compat import (
    Qwen3OmniMoeVisionEncoderCompat,
    VisionRotaryInputs,
    VisionSequenceMetadata,
)
from sglang_omni.platforms.interface import JointRopeInplaceKernel


class Qwen3OmniVisionEncoder(Qwen3OmniMoeVisionEncoderCompat):
    def __init__(
        self,
        config: Qwen3OmniMoeVisionEncoderConfig,
        *,
        joint_rope_kernel: JointRopeInplaceKernel | None,
    ) -> None:
        super().__init__(config)
        self.joint_rope_kernel: JointRopeInplaceKernel | None = joint_rope_kernel
        # note (yzxiao): Keep attention and its RoPE input format paired at construction.
        for block in self.blocks:
            block.attn = Qwen3OmniVisionAttention(
                block.attn, joint_rope_kernel=joint_rope_kernel
            )

    def prepare_position_embeddings(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | VisionRotaryInputs:
        if self.joint_rope_kernel is None:
            return cos, sin
        else:
            half_rotary_dimension = cos.shape[-1] // 2
            # note (yzxiao): Recomputing FP32 coefficients would change BF16 semantics.
            return VisionRotaryInputs(
                cos_sin_cache=torch.cat(
                    (
                        cos[:, :half_rotary_dimension].float(),
                        sin[:, :half_rotary_dimension].float(),
                    ),
                    dim=-1,
                ).contiguous(),
                positions=torch.arange(
                    cos.shape[0], device=cos.device, dtype=torch.int64
                ),
            )

    def prepare_attention_metadata(
        self,
        cumulative_sequence_lengths: torch.Tensor,
    ) -> dict[str, VisionSequenceMetadata]:
        # note (yzxiao): Reuse host lengths to avoid per-layer device synchronization.
        sequence_patch_counts = tuple(
            (
                cumulative_sequence_lengths[1:] - cumulative_sequence_lengths[:-1]
            ).tolist()
        )
        return {
            "sequence_metadata": VisionSequenceMetadata(
                sequence_patch_counts=sequence_patch_counts,
                max_sequence_patch_count=max(sequence_patch_counts),
            )
        }
