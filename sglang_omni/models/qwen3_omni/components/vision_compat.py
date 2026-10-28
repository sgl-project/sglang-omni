# SPDX-License-Identifier: Apache-2.0
"""Transformers compatibility boundary for the Qwen3-Omni vision encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sglang.srt.layers.attention.vision import VisionAttentionMetadata
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from transformers.vision_utils import get_vision_cu_seqlens, get_vision_position_ids


@dataclass(frozen=True, kw_only=True)
class VisionRotaryInputs:
    cos_sin_cache: torch.Tensor
    positions: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class VisionSequenceMetadata:
    sequence_patch_counts: tuple[int, ...]
    packed_attention_metadata: VisionAttentionMetadata | None


class Qwen3OmniMoeVisionEncoderCompat(hf_modeling.Qwen3OmniMoeVisionEncoder):
    """HF vision encoder with the bilinear position interpolation computed on its device."""

    def prepare_position_embeddings(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | VisionRotaryInputs:
        return cos, sin

    def prepare_attention_metadata(
        self,
        sequence_patch_counts: tuple[int, ...],
        cumulative_sequence_lengths: torch.Tensor,
    ) -> dict[str, VisionSequenceMetadata]:
        """The extra attention keywords every block receives."""
        return {}

    def interpolate_position_embeddings(
        self,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        """Bilinear position embeddings of every grid, in merge-block order, repeated per frame."""
        side = self.num_grid_per_side
        merge_size = self.spatial_merge_size
        hidden_size = self.pos_embed.embedding_dim
        device = self.pos_embed.weight.device
        # note (ratish): Transformers 5.6's arithmetic, kept bit for bit; the checkpoint scores lower with one-rounding
        # bilinear. fp32 linspace coordinates (CUDA's equal the CPU's), corner weights rounded to the table dtype,
        # the four weighted rows added in order in that dtype.
        embeddings = []
        for frames, height, width in grid_thw:
            rows = torch.linspace(
                0, side - 1, height, dtype=torch.float32, device=device
            )
            columns = torch.linspace(
                0, side - 1, width, dtype=torch.float32, device=device
            )
            rows_floor = rows.int()
            columns_floor = columns.int()
            rows_ceil = (rows_floor + 1).clip(max=side - 1)
            columns_ceil = (columns_floor + 1).clip(max=side - 1)
            row_fractions = rows - rows_floor
            column_fractions = columns - columns_floor
            row_bases = (rows_floor * side)[:, None]
            row_ceil_bases = (rows_ceil * side)[:, None]
            corner_indices = torch.stack(
                (
                    row_bases + columns_floor,
                    row_bases + columns_ceil,
                    row_ceil_bases + columns_floor,
                    row_ceil_bases + columns_ceil,
                )
            ).reshape(4, -1)
            corner_weights = (
                torch.stack(
                    (
                        (1 - row_fractions)[:, None] * (1 - column_fractions),
                        (1 - row_fractions)[:, None] * column_fractions,
                        row_fractions[:, None] * (1 - column_fractions),
                        row_fractions[:, None] * column_fractions,
                    )
                )
                .reshape(4, -1)
                .to(self.pos_embed.weight.dtype)
            )
            corners = self.pos_embed(corner_indices) * corner_weights[:, :, None]
            spatial = corners[0] + corners[1] + corners[2] + corners[3]
            merged = (
                spatial.reshape(
                    height // merge_size,
                    merge_size,
                    width // merge_size,
                    merge_size,
                    hidden_size,
                )
                .permute(0, 2, 1, 3, 4)
                .reshape(1, -1, hidden_size)
            )
            embeddings.append(merged.expand(frames, -1, -1).reshape(-1, hidden_size))
        return torch.cat(embeddings)

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | hf_modeling.BaseModelOutputWithDeepstackFeatures:
        """Run the HF encoder with the positional interpolation computed on its device."""
        # HF may pass its device-generated interpolation tensors through
        # kwargs. This compatibility path always recomputes them from the grid.
        kwargs.pop("interp_indices", None)
        kwargs.pop("interp_weights", None)
        # note (ratish): the rotary positions and attention lengths are derived on the host
        # from the grid; the device receives one copy of each and is never read back.
        host_grid_thw = grid_thw.cpu()
        position_ids = get_vision_position_ids(
            host_grid_thw, self.spatial_merge_size, kwargs=kwargs
        )
        host_cu_seqlens = get_vision_cu_seqlens(host_grid_thw, kwargs=kwargs)
        sequence_patch_counts = tuple(
            (host_cu_seqlens[1:] - host_cu_seqlens[:-1]).tolist()
        )
        max_seqlen = max(sequence_patch_counts)
        device = hidden_states.device
        cu_seqlens = host_cu_seqlens.to(device, non_blocking=True)
        position_ids = position_ids.to(device, non_blocking=True)
        attention_metadata_arguments = self.prepare_attention_metadata(
            sequence_patch_counts, cu_seqlens
        )

        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.interpolate_position_embeddings(host_grid_thw.tolist())
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
        cos, sin = self.rotary_pos_emb(hidden_states, position_ids)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        position_embeddings = self.prepare_position_embeddings(cos, sin)

        deepstack_feature_lists = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                **attention_metadata_arguments,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                merger_index = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[merger_index](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)
            else:
                pass

        merged_hidden_states = self.merger(hidden_states)

        return hf_modeling.BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
            deepstack_features=deepstack_feature_lists,
        )
