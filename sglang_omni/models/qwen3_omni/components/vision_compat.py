# SPDX-License-Identifier: Apache-2.0
"""Transformers compatibility boundary for the Qwen3-Omni vision encoder."""

from __future__ import annotations

import torch
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs


class Qwen3OmniMoeVisionEncoderCompat(hf_modeling.Qwen3OmniMoeVisionEncoder):
    """HF vision encoder with the Transformers 5.6 interpolation arithmetic."""

    def _legacy_pos_embed_interpolate(
        self,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        grid_thw_list = grid_thw.tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]
        device = self.pos_embed.weight.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        # Keep these coordinate and weight operations byte-for-byte equivalent
        # to the Transformers 5.6 implementation. In particular, linspace runs
        # on CPU and the materialized weights use the embedding table dtype.
        for _, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for corner in range(4):
                idx_list[corner].extend(indices[corner].tolist())
                weight_list[corner].extend(weights[corner].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(
            weight_list,
            dtype=self.pos_embed.weight.dtype,
            device=device,
        )
        corners = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = corners[0] + corners[1] + corners[2] + corners[3]

        patch_pos_embeds = patch_pos_embeds.split(
            [h * w for h, w in zip(grid_hs, grid_ws)]
        )

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(
                    t,
                    h // merge_size,
                    merge_size,
                    w // merge_size,
                    merge_size,
                    -1,
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute)

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | hf_modeling.BaseModelOutputWithDeepstackFeatures:
        """Run the HF 5.12.1 encoder with legacy positional interpolation."""
        # HF 5.12 may pass its device-generated interpolation tensors through
        # kwargs. This compatibility path always recomputes the legacy values.
        kwargs.pop("bilinear_indices", None)
        kwargs.pop("bilinear_weights", None)
        position_ids = hf_modeling.get_vision_position_ids(
            grid_thw, self.spatial_merge_size, kwargs=kwargs
        )
        cu_seqlens = hf_modeling.get_vision_cu_seqlens(grid_thw, kwargs=kwargs)

        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self._legacy_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
        rotary_pos_emb = self.rotary_pos_emb(position_ids)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        deepstack_feature_lists = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                merger_index = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[merger_index](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        merged_hidden_states = self.merger(hidden_states)

        return hf_modeling.BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
            deepstack_features=deepstack_feature_lists,
        )
